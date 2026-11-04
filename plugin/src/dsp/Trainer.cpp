#include "Trainer.hpp"

#include <modules/json/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

#include "Features.hpp"

namespace beat {

using json = nlohmann::json;

namespace {

const double STRETCH[] = {0.8, 0.9, 1.0, 1.12, 1.25};  // hop scale factors, as python/beatref/train.py
const double GAIN_DB[] = {6, -3, -12, -21, -30};        // one per stretch factor, spanning train.py's random gain

/** A dense layer is a convolution with one tap. Weights are [tap][in][out] (Keras order: the last
tap multiplies the current frame), so z[t] = b + sum_i W_i' x[t - (k - 1 - i) d]. */
struct Layer {
	int in = 0, out = 0, k = 1, d = 1;
	std::vector<float> w, b;

	float* tap(int i) { return &w[size_t(i) * in * out]; }
	const float* tap(int i) const { return &w[size_t(i) * in * out]; }

	void forward(const float* x, int T, float* z) const {
		for (int t = 0; t < T; t++) {
			float* zt = z + size_t(t) * out;
			std::copy(b.begin(), b.end(), zt);
			for (int i = 0; i < k; i++) {
				int tt = t - (k - 1 - i) * d;
				if (tt < 0)
					continue;
				const float* xt = x + size_t(tt) * in;
				const float* W = tap(i);
				for (int c = 0; c < in; c++) {
					float xv = xt[c];
					const float* wr = W + size_t(c) * out;
					for (int o = 0; o < out; o++)
						zt[o] += xv * wr[o];
				}
			}
		}
	}

	/** Accumulates dW and db; dx (if not null) gets the gradient with respect to x. */
	void backward(const float* x, const float* dz, int T, float* dW, float* db, float* dx) const {
		for (int t = 0; t < T; t++) {
			const float* g = dz + size_t(t) * out;
			for (int o = 0; o < out; o++)
				db[o] += g[o];
			for (int i = 0; i < k; i++) {
				int tt = t - (k - 1 - i) * d;
				if (tt < 0)
					continue;
				const float* xt = x + size_t(tt) * in;
				const float* W = tap(i);
				float* dWi = dW + size_t(i) * in * out;
				float* dxt = dx ? dx + size_t(tt) * in : nullptr;
				for (int c = 0; c < in; c++) {
					float xv = xt[c], acc = 0;
					const float* wr = W + size_t(c) * out;
					float* dwr = dWi + size_t(c) * out;
					for (int o = 0; o < out; o++) {
						dwr[o] += xv * g[o];
						acc += wr[o] * g[o];
					}
					if (dxt)
						dxt[c] += acc;
				}
			}
		}
	}
};

inline float elu(float x) {
	return x > 0 ? x : std::expm1(x);
}

/** Rounds through text, which Rack's fast-math flags can't turn into a multiply by an inexact 0.1. */
double rounded(double v, const char* format) {
	char s[32];
	std::snprintf(s, sizeof s, format, v);
	return std::strtod(s, nullptr);
}

double round8(double v) {
	return rounded(v, "%.8g");
}

const int GRID = 200;  // training crops start every 2 s

/** Part of a recording at one stretch factor: the frozen layers' output and the frame targets.
Crops start cold, as python/beatref/train.py's do, so the frozen output is also kept as it is
when the features start at each grid point: it differs from the warm one for `reach` frames. */
struct Item {
	int n = 0, width = 0, reach = 0;
	std::vector<float> prefix;   // n x width
	std::vector<float> cold;     // per grid point, reach x width
	std::vector<float> target;   // n x 2 (beat, downbeat)
	bool hasDown = false;
};

/** numpy.percentile's default (linear) interpolation. */
double percentile(std::vector<double> v, double q) {
	std::sort(v.begin(), v.end());
	double pos = q / 100 * (v.size() - 1);
	size_t i = size_t(pos);
	return i + 1 < v.size() ? v[i] + (pos - i) * (v[i + 1] - v[i]) : v.back();
}

uint32_t u32(const unsigned char* p) {
	return p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24;
}

} // namespace

Recording Recording::load(const std::string& stem) {
	Recording r;
	r.name = stem.substr(stem.find_last_of("/\\") + 1);
	std::ifstream in(stem + ".wav", std::ios::binary);
	if (!in)
		throw std::runtime_error("can't open " + stem + ".wav");
	std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(&bytes[8], "WAVE", 4))
		throw std::runtime_error(r.name + ".wav is not a WAV file");
	int format = 0, channels = 0, bits = 0;
	const unsigned char* data = nullptr;
	size_t dataSize = 0;
	for (size_t p = 12; p + 8 <= bytes.size();) {
		size_t size = u32(&bytes[p + 4]);
		const unsigned char* body = &bytes[p + 8];
		size = std::min(size, bytes.size() - p - 8);  // a recording cut short still reads
		if (!std::memcmp(&bytes[p], "fmt ", 4) && size >= 16) {
			format = body[0] | body[1] << 8;
			channels = body[2] | body[3] << 8;
			r.sampleRate = u32(body + 4);
			bits = body[14] | body[15] << 8;
			if (format == 0xFFFE && size >= 26)
				format = body[24] | body[25] << 8;  // WAVE_FORMAT_EXTENSIBLE: the subformat
		}
		else if (!std::memcmp(&bytes[p], "data", 4)) {
			data = body;
			dataSize = size;
		}
		p += 8 + size + (size & 1);
	}
	bool pcm = format == 1 && (bits == 16 || bits == 24 || bits == 32), flt = format == 3 && bits == 32;
	if (!data || channels < 1 || r.sampleRate <= 0 || !(pcm || flt))
		throw std::runtime_error(r.name + ".wav: only 16, 24 and 32-bit PCM and 32-bit float are supported");
	int bytesPer = bits / 8;
	size_t frames = dataSize / (size_t(bytesPer) * channels);
	r.audio.resize(frames);
	for (size_t i = 0; i < frames; i++) {
		double sum = 0;
		for (int c = 0; c < channels; c++) {
			const unsigned char* s = data + (i * channels + c) * bytesPer;
			if (flt) {
				float f;
				std::memcpy(&f, s, 4);
				sum += f;
			}
			else if (bits == 16)
				sum += int16_t(s[0] | s[1] << 8) / 32768.0;
			else if (bits == 24)
				sum += int32_t(uint32_t(s[0] << 8 | s[1] << 16 | s[2] << 24)) / 2147483648.0;
			else
				sum += int32_t(u32(s)) / 2147483648.0;
		}
		r.audio[i] = float(sum / channels);
	}

	std::ifstream grid(stem + ".beats");
	if (!grid)
		throw std::runtime_error("can't open " + stem + ".beats");
	std::string line;
	bool numbered = true;
	while (std::getline(grid, line)) {
		std::istringstream fields(line);
		double t;
		int n;
		if (!(fields >> t))
			continue;
		r.beats.push_back(t);
		if (fields >> n)
			r.numbers.push_back(n);
		else
			numbered = false;
	}
	if (!numbered)
		r.numbers.clear();
	if (r.beats.size() < 2)
		throw std::runtime_error(r.name + ".beats has fewer than two beats");
	return r;
}

struct Trainer::Impl {
	json spec;
	std::string baseName;
	Tune tune;
	Frontend fe;
	int delay = 1;
	std::vector<Layer> layers;
	int first = 0;  // index of the first trainable layer
	std::vector<Item> train, val;
	std::mt19937_64 rng;

	int top() const { return int(layers.size()) - first; }

	std::vector<float> frozen(const float* x, int T) const {
		std::vector<float> in(x, x + size_t(T) * layers[0].in);
		for (int l = 0; l < first; l++) {
			std::vector<float> z(size_t(T) * layers[l].out);
			layers[l].forward(in.data(), T, z.data());
			for (float& v : z)
				v = elu(v);
			in.swap(z);
		}
		return in;
	}

	Item item(const std::vector<float>& x, const std::vector<float>& target, bool hasDown, int s, int e) const {
		Item it;
		int nf = layers[0].in;
		it.n = e - s;
		it.width = layers[first].in;
		it.hasDown = hasDown;
		it.prefix = frozen(&x[size_t(s) * nf], it.n);
		it.reach = 1;
		for (int l = 0; l < first; l++)
			it.reach += (layers[l].k - 1) * layers[l].d;
		for (int b = 0; b < it.n; b += GRID) {
			std::vector<float> c = frozen(&x[size_t(s + b) * nf], std::min(it.reach, it.n - b));
			c.resize(size_t(it.reach) * it.width);
			it.cold.insert(it.cold.end(), c.begin(), c.end());
		}
		it.target.assign(target.begin() + size_t(s) * 2, target.begin() + size_t(e) * 2);
		return it;
	}

	/** Features and targets of a recording at one stretch factor, split into training and held-back items. */
	std::pair<Item, Item> prepare(const Recording& rec, double stretch, double gainDb) const {
		Features f(rec.sampleRate, fe, 100.0 / stretch);
		std::vector<float> x;
		double gain = std::pow(10.0, gainDb / 20);
		for (float s : rec.audio)
			if (f.push(s * gain))
				x.insert(x.end(), f.values(), f.values() + f.size());
		int n = int(x.size() / f.size());
		std::vector<float> target(size_t(n) * 2, 0.f);
		bool hasDown = !rec.numbers.empty();
		double framesPerS = rec.sampleRate / f.hopSize();
		for (size_t j = 0; j < rec.beats.size(); j++) {
			int k = int(std::floor(rec.beats[j] * framesPerS)) + delay;
			bool down = hasDown && rec.numbers[j] == 1;
			for (int off : {-1, 1, 0}) {  // widened: neighbours count half
				int i = k + off;
				if (i < 0 || i >= n)
					continue;
				float v = off ? 0.5f : 1.f;
				target[size_t(i) * 2] = std::max(target[size_t(i) * 2], v);
				if (down)
					target[size_t(i) * 2 + 1] = std::max(target[size_t(i) * 2 + 1], v);
			}
		}
		int k = int(n * (1 - tune.holdout));
		return {item(x, target, hasDown, 0, k), item(x, target, hasDown, k, n)};
	}

	struct Crop {
		const Item* item;
		int s, T;
	};

	std::vector<Crop> sample(const std::vector<Item>& items, std::mt19937_64& g) {
		std::vector<Crop> b;
		for (int i = 0; i < tune.batch; i++) {
			const Item& it = items[g() % items.size()];
			int s = GRID * int(g() % uint64_t(std::max(it.n - tune.crop, 0) / GRID + 1));
			b.push_back({&it, s, std::min(tune.crop, it.n - s)});
		}
		return b;
	}

	/** Masked BCE over the batch, with gradients into `grads` (one vector per trainable layer:
	weights then bias) unless it is null; dropout only when training. */
	double step(const std::vector<Crop>& batch, std::vector<std::vector<float>>* grads) {
		double count = 0;
		for (const Crop& c : batch)
			count += c.T * (c.item->hasDown ? 2 : 1);
		double loss = 0;
		bool training = grads != nullptr;
		float keep = float(1 - tune.dropout), scale = 1 / keep;
		std::uniform_real_distribution<float> u(0, 1);
		int L = top();
		for (const Crop& c : batch) {
			int T = c.T;
			std::vector<std::vector<float>> h(L), a(L), m(L);  // inputs to each layer; pre-dropout ELU outputs; dropout masks
			const Item& it = *c.item;
			h[0].assign(it.prefix.begin() + size_t(c.s) * it.width, it.prefix.begin() + size_t(c.s + T) * it.width);
			const float* cold = &it.cold[size_t(c.s / GRID) * it.reach * it.width];
			std::copy(cold, cold + size_t(std::min(it.reach, T)) * it.width, h[0].begin());
			auto drop = [&](std::vector<float>& v, std::vector<float>& mask) {
				if (!training)
					return;
				mask.resize(v.size());
				for (size_t i = 0; i < v.size(); i++)
					v[i] *= mask[i] = u(rng) < keep ? scale : 0.f;
			};
			std::vector<float> m0;
			drop(h[0], m0);
			for (int l = 0; l + 1 < L; l++) {
				const Layer& ly = layers[first + l];
				a[l].resize(size_t(T) * ly.out);
				ly.forward(h[l].data(), T, a[l].data());
				for (float& v : a[l])
					v = elu(v);
				h[l + 1] = a[l];
				drop(h[l + 1], m[l]);
			}
			const Layer& outL = layers.back();
			std::vector<float> y(size_t(T) * outL.out), dy(y.size(), 0.f);
			outL.forward(h[L - 1].data(), T, y.data());
			const float* tg = &c.item->target[size_t(c.s) * 2];
			for (int t = 0; t < T; t++)
				for (int o = 0; o < outL.out && o < 2; o++) {
					if (o == 1 && !c.item->hasDown)
						continue;
					float z = y[size_t(t) * outL.out + o], target = tg[t * 2 + o];
					loss += std::max(z, 0.f) - z * target + std::log1p(std::exp(-std::fabs(z)));
					dy[size_t(t) * outL.out + o] = float((1 / (1 + std::exp(-z)) - target) / count);
				}
			if (!training)
				continue;
			std::vector<float> dh(size_t(T) * outL.in, 0.f);
			outL.backward(h[L - 1].data(), dy.data(), T, (*grads)[2 * (L - 1)].data(), (*grads)[2 * (L - 1) + 1].data(), dh.data());
			for (int l = L - 2; l >= 0; l--) {
				const Layer& ly = layers[first + l];
				for (size_t i = 0; i < dh.size(); i++)  // through dropout, then ELU (its slope is a + 1 below zero)
					dh[i] *= m[l][i] * (a[l][i] > 0 ? 1.f : a[l][i] + 1.f);
				std::vector<float> dx(l > 0 ? size_t(T) * ly.in : 0, 0.f);
				ly.backward(h[l].data(), dh.data(), T, (*grads)[2 * l].data(), (*grads)[2 * l + 1].data(), l > 0 ? dx.data() : nullptr);
				dh.swap(dx);
			}
		}
		return loss / count;
	}

};

Trainer::Trainer(const std::string& modelJson, const std::string& baseName, const Tune& tune) : impl(new Impl) {
	Impl& m = *impl;
	m.baseName = baseName;
	m.tune = tune;
	m.rng.seed(tune.seed);
	try {
		m.spec = json::parse(modelJson);
	}
	catch (const std::exception& e) {
		delete impl;
		throw std::runtime_error(baseName + ": not JSON (" + e.what() + ")");
	}
	auto fail = [&](const std::string& why) {
		delete impl;
		throw std::runtime_error(baseName + ": " + why);
	};
	json meta = m.spec.value("beatref", json::object());
	json fe = meta.value("frontend", json::object());
	m.fe.bandsPerOctave = fe.value("bands_per_octave", m.fe.bandsPerOctave);
	m.fe.fmin = fe.value("fmin", m.fe.fmin);
	m.fe.fmax = fe.value("fmax", m.fe.fmax);
	m.fe.logMul = fe.value("log_mul", m.fe.logMul);
	m.fe.diff = fe.value("diff", m.fe.diff);
	m.delay = meta.value("hyper", json::object()).value("delay", 1);
	if (!m.spec.contains("layers") || !m.spec["layers"].is_array())
		fail("no layers");
	int width = m.fe.size();
	const json& ls = m.spec["layers"];
	for (size_t i = 0; i < ls.size(); i++) {
		const json& j = ls[i];
		std::string type = j.value("type", ""), act = j.value("activation", "");
		bool last = i + 1 == ls.size();
		if ((type != "dense" && type != "conv1d") || act != (last ? "sigmoid" : "elu") || j.value("groups", 1) != 1)
			fail("learning mode adapts models made by beatref train (dense and conv1d layers with ELU, sigmoid outputs)");
		Layer ly;
		const json& w = j["weights"][0];
		if (type == "conv1d") {
			ly.k = int(w.size());
			ly.d = j["dilation"].is_array() ? j["dilation"].back().get<int>() : j["dilation"].get<int>();
		}
		const json& taps = type == "conv1d" ? w : json::array({w});
		ly.in = int(taps[0].size());
		ly.out = int(taps[0][0].size());
		if (ly.in != width)
			fail("layer " + std::to_string(i) + " takes " + std::to_string(ly.in) + " inputs, not " + std::to_string(width));
		for (const json& tap : taps)
			for (const json& row : tap)
				for (const json& v : row)
					ly.w.push_back(v.get<float>());
		for (const json& v : j["weights"][1])
			ly.b.push_back(v.get<float>());
		if (ly.w.size() != size_t(ly.k) * ly.in * ly.out || int(ly.b.size()) != ly.out)
			fail("layer " + std::to_string(i) + " has malformed weights");
		width = ly.out;
		m.layers.push_back(std::move(ly));
	}
	if (m.layers.size() < 2 || m.layers.back().out > 2)
		fail("expected a beat (and downbeat) output layer");
	int top = tune.layers > 0 ? std::min(tune.layers, int(m.layers.size()) - 1) : int(m.layers.size()) - 1;
	m.first = int(m.layers.size()) - top;
}

Trainer::~Trainer() {
	delete impl;
}

std::string Trainer::run(const std::vector<Recording>& recordings, const std::atomic<bool>& cancel,
	const std::function<void(const Progress&)>& report) {
	Impl& m = *impl;
	const Tune& t = m.tune;
	auto say = [&](const std::string& stage) {
		prog.stage = stage;
		if (report)
			report(prog);
	};
	if (recordings.empty())
		throw std::runtime_error("no recordings");

	m.train.clear();
	m.val.clear();
	double minutes = 0;
	for (size_t r = 0; r < recordings.size(); r++) {
		say("Analysing recording " + std::to_string(r + 1) + " of " + std::to_string(recordings.size()));
		for (int i = 0; i < 5; i++) {
			if (cancel)
				return "";
			double s = STRETCH[i];
			auto items = m.prepare(recordings[r], s, GAIN_DB[(i + r) % 5]);
			if (s == 1.0)
				minutes += items.first.n / 6000.0;
			if (items.first.n > 0 && items.second.n > 0) {
				m.train.push_back(std::move(items.first));
				m.val.push_back(std::move(items.second));
			}
		}
	}

	if (m.train.empty())
		throw std::runtime_error("the recordings are too short");
	int L = m.top();
	std::vector<std::vector<float>> anchor, grads, mom, vel;
	for (int l = 0; l < L; l++)
		for (std::vector<float>* p : {&m.layers[m.first + l].w, &m.layers[m.first + l].b}) {
			anchor.push_back(*p);
			grads.emplace_back(p->size());
			mom.emplace_back(p->size(), 0.f);
			vel.emplace_back(p->size(), 0.f);
		}
	auto param = [&](size_t i) -> std::vector<float>& {
		Layer& ly = m.layers[m.first + i / 2];
		return i % 2 ? ly.b : ly.w;
	};

	std::mt19937_64 vrng(1234);
	std::vector<std::vector<Impl::Crop>> vbatches;
	for (int i = 0; i < 8; i++)
		vbatches.push_back(m.sample(m.val, vrng));
	auto validate = [&]() {
		double sum = 0;
		for (auto& b : vbatches)
			sum += m.step(b, nullptr);
		return sum / vbatches.size();
	};

	prog.epochs = t.epochs;
	prog.baseLoss = prog.bestLoss = prog.lastLoss = validate();
	std::vector<std::vector<float>> best;
	int total = t.epochs * t.steps, n = 0;
	const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
	for (int epoch = 0; epoch < t.epochs; epoch++) {
		prog.epoch = epoch + 1;
		say("Adapting");
		for (int s = 0; s < t.steps; s++, n++) {
			if (cancel)
				return "";
			for (auto& g : grads)
				std::fill(g.begin(), g.end(), 0.f);
			m.step(m.sample(m.train, m.rng), &grads);
			double norm = 0;
			for (size_t i = 0; i < grads.size(); i++) {
				std::vector<float>& p = param(i);
				for (size_t j = 0; j < p.size(); j++) {
					grads[i][j] += float(2 * t.l2sp * (p[j] - anchor[i][j]));
					norm += double(grads[i][j]) * grads[i][j];
				}
			}
			float clip = float(std::min(1.0, 1.0 / (std::sqrt(norm) + 1e-6)));
			double lr = t.lr * 0.5 * (1 + std::cos(std::acos(-1.0) * n / total));
			double c1 = 1 - std::pow(b1, n + 1), c2 = 1 - std::pow(b2, n + 1);
			for (size_t i = 0; i < grads.size(); i++) {
				std::vector<float>& p = param(i);
				for (size_t j = 0; j < p.size(); j++) {
					float g = grads[i][j] * clip;
					mom[i][j] = float(b1 * mom[i][j] + (1 - b1) * g);
					vel[i][j] = float(b2 * vel[i][j] + (1 - b2) * g * g);
					p[j] -= float(lr * (mom[i][j] / c1) / (std::sqrt(vel[i][j] / c2) + eps));
				}
			}
		}
		prog.lastLoss = validate();
		if (prog.lastLoss < prog.bestLoss) {
			prog.bestLoss = prog.lastLoss;
			best.clear();
			for (size_t i = 0; i < grads.size(); i++)
				best.push_back(param(i));
		}
		say("Validated");
	}
	if (best.empty()) {
		say("No epoch beat the base model on the held-back audio");
		return "";
	}

	json out = m.spec;
	for (int l = 0; l < L; l++) {
		const Layer& ly = m.layers[m.first + l];
		const std::vector<float>& w = best[2 * l];
		json taps = json::array();
		for (int i = 0; i < ly.k; i++) {
			json rows = json::array();
			for (int c = 0; c < ly.in; c++) {
				json row = json::array();
				for (int o = 0; o < ly.out; o++)
					row.push_back(round8(w[(size_t(i) * ly.in + c) * ly.out + o]));
				rows.push_back(row);
			}
			taps.push_back(rows);
		}
		json bias = json::array();
		for (float v : best[2 * l + 1])
			bias.push_back(round8(v));
		json& j = out["layers"][m.first + l];
		j["weights"][0] = j.value("type", "") == "conv1d" ? taps : taps[0];
		j["weights"][1] = bias;
	}

	std::vector<double> bpm;
	std::map<int, int> meters;
	for (const Recording& r : recordings) {
		for (size_t i = 1; i < r.beats.size(); i++)
			if (r.beats[i] > r.beats[i - 1])
				bpm.push_back(60 / (r.beats[i] - r.beats[i - 1]));
		for (size_t i = 0; i + 1 < r.numbers.size(); i++)
			if (r.numbers[i + 1] == 1)
				meters[r.numbers[i]]++;
	}
	json& meta = out["beatref"];
	double lo = rounded(percentile(bpm, 5) / 1.05, "%.1f"), hi = rounded(percentile(bpm, 95) * 1.05, "%.1f");
	meta["suggest"] = {{"min_bpm", std::min(60.0, lo)}, {"max_bpm", std::max(200.0, hi)}, {"pref_min_bpm", lo}, {"pref_max_bpm", hi}};
	if (!meters.empty())
		meta["suggest"]["beats_per_bar"] = std::max_element(meters.begin(), meters.end(),
			[](const auto& a, const auto& b) { return a.second < b.second; })->first;
	meta["adapted"] = {{"base", m.baseName}, {"recordings", recordings.size()}, {"minutes", rounded(minutes, "%.2f")},
		{"trainer", "rack"}, {"tune", {{"layers", t.layers}, {"epochs", t.epochs}, {"steps", t.steps}, {"lr", t.lr},
			{"l2sp", t.l2sp}, {"holdout", t.holdout}, {"crop", t.crop}, {"seed", t.seed}}},
		{"loss", {{"base", prog.baseLoss}, {"adapted", prog.bestLoss}}}};
	say("Done");
	return out.dump();
}

} // namespace beat

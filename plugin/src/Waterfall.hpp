#pragma once
#include <deque>

#include "BeatDetector.hpp"

/** The scrolling display: band-coloured envelope, onset ticks and the beat grid on one time axis,
with the past left of the now line and predicted beats to its right. The audio thread only pushes
frame summaries into a ring buffer; everything here runs on the UI thread. */
struct Waterfall : widget::Widget {
	BeatDetector* module = nullptr;

	struct Mark {
		double time;
		float strength;
		bool bar, locked;
	};
	std::deque<beat::Frame> history;
	std::deque<Mark> onsets, beats;
	beat::Frame last;
	float act1 = 0, act2 = 0;  // the two previous activations, for peak picking
	float peak = 1e-3f;        // envelope auto-gain
	bool fresh = true;

	int image = -1, imgW = 0, imgH = 0;
	std::vector<uint8_t> pixels;

	~Waterfall() {
		if (image >= 0 && APP && APP->window)
			nvgDeleteImage(APP->window->vg, image);
	}

	float zoom() const { return BeatDetector::ZOOMS[module ? module->zoom : 1]; }

	void ingest(const beat::Frame& f, float latency) {
		history.push_back(f);
		peak = std::max({f.env, peak * 0.9995f, 1e-4f});
		if (act1 > act2 && act1 >= f.act && act1 > 0.3f)  // the previous frame was an onset peak
			onsets.push_back({last.time - latency, act1, false, false});
		act2 = act1;
		act1 = f.act;
		if (f.beatAt >= 0)
			beats.push_back({f.time - f.beatAt, 1, f.bar, f.locked});
		last = f;
		double keep = f.time - 20;
		while (!history.empty() && history.front().time < keep)
			history.pop_front();
		while (!onsets.empty() && onsets.front().time < keep)
			onsets.pop_front();
		while (!beats.empty() && beats.front().time < keep)
			beats.pop_front();
		fresh = true;
	}

	void step() override {
		if (module) {
			float latency = module->params[BeatDetector::LATENCY_PARAM].getValue() / 1000;
			while (!module->frames.empty())
				ingest(module->frames.shift(), latency);
		}
		else if (history.empty())
			demo();
		Widget::step();
	}

	/** A plausible groove for the module browser, where there is no module. */
	void demo() {
		const double bpm = 124, spb = 60 / bpm;
		for (int i = 0; i < 1200; i++) {
			beat::Frame f;
			f.time = i * 0.01;
			double b = f.time / spb, kick = b - std::floor(b), hat = std::fmod(b + 0.5, 1.0);
			double k = std::exp(-kick * spb / 0.09), h = std::exp(-hat * spb / 0.03), s = std::fmod(b, 2.0) >= 1 ? k : 0;
			f.env = 0.03 + 0.6 * k + 0.2 * h + 0.35 * s;
			f.bands[0] = 0.05 + 3 * k;
			f.bands[1] = 0.2 + 1.2 * s + 0.2 * h;
			f.bands[2] = 0.1 + 1.5 * h + 0.6 * s;
			f.act = std::max(k, 0.6 * h);
			f.bpm = bpm;
			f.conf = 0.93;
			f.locked = true;
			f.running = true;
			f.bps = bpm / 60;
			f.outPhase = b;
			int n = int(std::floor(b));
			f.beatAt = (b - n) * spb < 0.01 ? float((b - n) * spb) : -1;
			f.bar = n % 4 == 0;
			ingest(f, 0.0f);
		}
	}

	NVGcolor columnColour(const float bands[3], float level) const {
		static const float warm[3] = {1.0f, 0.42f, 0.12f}, mid[3] = {0.35f, 0.95f, 0.4f}, high[3] = {0.3f, 0.6f, 1.0f};
		float rgb[3];
		if (module && module->scheme == BeatDetector::SCHEME_MONO)
			rgb[0] = 0.82f, rgb[1] = 0.85f, rgb[2] = 0.9f;
		else {
			float sum = bands[0] + bands[1] + bands[2] + 1e-12f, top = 1e-6f;
			for (int c = 0; c < 3; c++) {
				rgb[c] = 0;
				for (int b = 0; b < 3; b++) {
					const float* tint = b == 0 ? warm : b == 1 ? mid : high;
					rgb[c] += std::sqrt(bands[b] / sum) * tint[c];
				}
				top = std::max(top, rgb[c]);
			}
			for (float& v : rgb)
				v /= top;
		}
		return nvgRGBf(rgb[0] * level, rgb[1] * level, rgb[2] * level);
	}

	/** Render the envelope left of the now line into the image, one pixel column per time bin. */
	void render(NVGcontext* vg, float nowX) {
		int w = std::max(1, int(nowX)), h = std::max(1, int(box.size.y * 2));
		if (w != imgW || h != imgH || image < 0) {
			if (image >= 0)
				nvgDeleteImage(vg, image);
			imgW = w, imgH = h;
			pixels.assign(size_t(w) * h * 4, 0);
			image = nvgCreateImageRGBA(vg, w, h, 0, pixels.data());
		}
		std::fill(pixels.begin(), pixels.end(), 0);
		double spc = zoom() / box.size.x, now = last.time;
		auto it = history.rbegin();
		for (int x = w - 1; x >= 0 && it != history.rend(); x--) {
			double start = now - (w - x) * spc;
			float env = 0, bands[3] = {};
			for (; it != history.rend() && it->time > start; ++it) {
				env = std::max(env, it->env);
				for (int b = 0; b < 3; b++)
					bands[b] += it->bands[b];
			}
			if (env <= 0)
				continue;
			float half = std::min(1.f, std::sqrt(env / peak)) * 0.42f * h;
			NVGcolor c = columnColour(bands, 1);
			for (int y = 0; y < h; y++) {
				float d = std::abs(y + 0.5f - h * 0.5f);
				float a = std::clamp(half - d + 0.5f, 0.f, 1.f);
				if (a <= 0)
					continue;
				float shade = 1 - 0.4f * d / std::max(half, 1.f);
				uint8_t* p = &pixels[(size_t(y) * w + x) * 4];
				p[0] = uint8_t(255 * c.r * shade);
				p[1] = uint8_t(255 * c.g * shade);
				p[2] = uint8_t(255 * c.b * shade);
				p[3] = uint8_t(255 * a);
			}
		}
		nvgUpdateImage(vg, image, pixels.data());
		fresh = false;
	}

	void line(NVGcontext* vg, float x, float y0, float y1, NVGcolor c, float width) {
		nvgBeginPath(vg);
		nvgMoveTo(vg, x, y0);
		nvgLineTo(vg, x, y1);
		nvgStrokeColor(vg, c);
		nvgStrokeWidth(vg, width);
		nvgStroke(vg);
	}

	void draw(const DrawArgs& args) override {
		nvgBeginPath(args.vg);
		nvgRoundedRect(args.vg, 0, 0, box.size.x, box.size.y, 3);
		nvgFillColor(args.vg, nvgRGB(0x12, 0x14, 0x17));
		nvgFill(args.vg);
	}

	void drawLayer(const DrawArgs& args, int layer) override {
		if (layer == 1)
			drawContent(args.vg);
		Widget::drawLayer(args, layer);
	}

	void drawContent(NVGcontext* vg) {
		float W = box.size.x, H = box.size.y, nowX = std::round(W * 2 / 3);
		double pps = W / zoom(), now = last.time;
		auto xOf = [&](double t) { return float(nowX - (now - t) * pps); };
		nvgSave(vg);
		nvgScissor(vg, 0, 0, W, H);

		// centre line, then the envelope image
		line(vg, 0, H / 2, W, nvgRGBA(255, 255, 255, 18), 1);
		if (fresh || image < 0)
			render(vg, nowX);
		nvgBeginPath(vg);
		nvgRect(vg, 0, 0, imgW, H);
		nvgFillPaint(vg, nvgImagePattern(vg, 0, 0, imgW, H, 0, image, 1));
		nvgFill(vg);

		// onsets: ticks up from the bottom, sized by strength
		for (const Mark& m : onsets) {
			float x = xOf(m.time);
			if (x >= 0 && x <= nowX)
				line(vg, x, H, H - 3 - 7 * m.strength, nvgRGBA(255, 255, 255, 110), 1);
		}

		// beat grid: past beats, then predicted ones right of the now line
		auto gridLine = [&](float x, bool bar, bool locked, bool future) {
			float a = (bar ? 0.85f : 0.4f) * (locked ? 1 : 0.35f) * (future ? 0.6f : 1);
			NVGcolor c = nvgRGBAf(1, 1, 1, a);
			if (future) {
				line(vg, x, 0, H * 0.22f, c, bar ? 2 : 1);
				line(vg, x, H * 0.78f, H, c, bar ? 2 : 1);
			}
			else
				line(vg, x, 0, H, c, bar ? 2 : 1);
		};
		for (const Mark& m : beats) {
			float x = xOf(m.time);
			if (x >= -2 && x <= nowX)
				gridLine(x, m.bar, m.locked, false);
		}
		if (last.running && last.bps > 0) {
			int bpb = std::max(1, last.beatsPerBar);
			for (double k = std::floor(last.outPhase) + 1;; k++) {
				float x = nowX + float((k - last.outPhase) / last.bps * pps);
				if (x > W + 2)
					break;
				double inBar = k - last.barOrigin;
				gridLine(x, inBar - bpb * std::floor(inBar / bpb) < 0.5, last.locked, true);
			}
		}
		line(vg, nowX, 0, H, nvgRGBA(255, 214, 120, 200), 1.25f);

		drawHud(vg, W, H);
		nvgResetScissor(vg);
		nvgRestore(vg);
	}

	void drawHud(NVGcontext* vg, float W, float H) {
		std::shared_ptr<window::Font> mono = APP->window->loadFont(asset::system("res/fonts/ShareTechMono-Regular.ttf"));
		std::shared_ptr<window::Font> sans = APP->window->loadFont(asset::system("res/fonts/Nunito-Bold.ttf"));
		if (!mono || !sans)
			return;
		NVGcolor text = nvgRGBA(240, 242, 245, 235), dim = nvgRGBA(240, 242, 245, 120);
		bool showing = last.running || last.locked;

		// a soft shade under the readouts, so they stay legible over the waveform
		nvgBeginPath(vg);
		nvgRect(vg, 0, 0, W, 34);
		nvgFillPaint(vg, nvgLinearGradient(vg, 0, 0, 0, 34, nvgRGBA(18, 20, 23, 200), nvgRGBA(18, 20, 23, 0)));
		nvgFill(vg);

		// BPM readout
		nvgFontFaceId(vg, mono->handle);
		nvgFontSize(vg, 20);
		nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
		nvgFillColor(vg, showing ? text : dim);
		std::string bpm = showing ? string::f("%5.1f", last.running ? last.bps * 60 : last.bpm) : "---.-";
		float bx = nvgText(vg, 6, 4, bpm.c_str(), nullptr);
		nvgFontFaceId(vg, sans->handle);
		nvgFontSize(vg, 8);
		nvgFillColor(vg, dim);
		nvgText(vg, bx + 3, 6, "BPM", nullptr);

		// position in the bar, as dots
		int bpb = std::clamp(last.beatsPerBar, 1, 16);
		double inBar = last.outPhase - last.barOrigin;
		int current = int(inBar - bpb * std::floor(inBar / bpb));
		for (int i = 0; i < bpb; i++) {
			nvgBeginPath(vg);
			nvgCircle(vg, 8 + i * 6.5f, 28, 2.1f);
			nvgFillColor(vg, last.running && i == current ? text : nvgRGBA(255, 255, 255, 45));
			nvgFill(vg);
		}

		// lock state
		const char* state = last.held ? "HOLD" : last.locked ? "LOCKED" : last.running ? "COASTING" : "LISTENING";
		NVGcolor led = last.locked ? nvgRGB(0x4c, 0xd9, 0x64) : last.running ? nvgRGB(0xff, 0xb3, 0x40) : nvgRGB(0x55, 0x5a, 0x60);
		nvgFontSize(vg, 8);
		nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_TOP);
		nvgFillColor(vg, text);
		float tw = nvgTextBounds(vg, 0, 0, state, nullptr, nullptr);
		nvgText(vg, W - 6, 5, state, nullptr);
		nvgBeginPath(vg);
		nvgCircle(vg, W - 11 - tw, 9, 2.5f);
		nvgFillColor(vg, led);
		nvgFill(vg);

		// confidence meter with the lock threshold
		float mw = 46, mx = W - 6 - mw, my = 17;
		nvgBeginPath(vg);
		nvgRoundedRect(vg, mx, my, mw, 3, 1.5f);
		nvgFillColor(vg, nvgRGBA(255, 255, 255, 30));
		nvgFill(vg);
		nvgBeginPath(vg);
		nvgRoundedRect(vg, mx, my, mw * std::clamp(last.conf, 0.f, 1.f), 3, 1.5f);
		nvgFillColor(vg, nvgRGBA(255, 255, 255, 170));
		nvgFill(vg);
		float threshold = module ? module->params[BeatDetector::LOCK_THRESHOLD_PARAM].getValue() : 0.6f;
		line(vg, mx + mw * threshold, my - 1.5f, my + 4.5f, led, 1);
	}
};

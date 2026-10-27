#include "Features.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>

namespace beat {

static const double REF_SR = 44100;
static const int REF_WIN = 2048;

std::vector<std::array<int, 3>> Frontend::triangles() const {
	// centre bins round half to even, as numpy does
	std::fesetround(FE_TONEAREST);
	const int nbins = REF_WIN / 2 + 1;
	const double bpo = bandsPerOctave;
	std::vector<int> c;
	for (double k = std::floor(bpo * std::log2(fmin / 440)); k <= std::ceil(bpo * std::log2(fmax / 440)); k++)
		c.push_back(std::clamp(int(std::nearbyint(440 * std::pow(2.0, k / bpo) * REF_WIN / REF_SR)), 1, nbins - 1));
	c.erase(std::unique(c.begin(), c.end()), c.end());
	std::vector<std::array<int, 3>> t;
	for (size_t j = 0; j + 2 < c.size(); j++)
		t.push_back({c[j], c[j + 1], c[j + 2]});
	return t;
}

Features::Features(double sr, const Frontend& frontend, double fps)
	: fe(frontend),
	  win(int(std::nearbyint(REF_WIN * sr / REF_SR))),
	  nfft(1 << int(std::ceil(std::log2(win)))),
	  hop(int(std::nearbyint(sr / fps))),
	  fft(nfft),
	  ring(win),
	  frame(nfft),
	  window(win),
	  mag(nfft / 2 + 1),
	  spec(nfft / 2 + 1) {
	double sum = 0;
	for (int i = 0; i < win; i++)
		sum += window[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / win);
	for (double& w : window)
		w *= 2 / sum;  // a full-scale sinusoid has magnitude 1

	// The reference filterbank, applied to this grid's spectrum linearly interpolated at the
	// reference bin frequencies, collapsed into one sparse filter per band.
	const int nref = REF_WIN / 2 + 1, top = nfft / 2;
	for (auto [a, c, b] : fe.triangles()) {
		std::vector<double> ref(nref), dense(top + 1);
		double area = 0;
		for (int k = a; k <= b; k++)
			area += ref[k] = std::max(0.0, std::min(double(k - a) / (c - a), double(b - k) / (b - c)));
		for (int k = a; k <= b; k++) {
			double p = k * REF_SR / REF_WIN * nfft / sr;
			if (p > top)
				continue;
			int lo = int(std::floor(p));
			dense[lo] += (1 - (p - lo)) * ref[k] / area;
			dense[std::min(lo + 1, top)] += (p - lo) * ref[k] / area;
		}
		std::vector<Tap> taps;
		for (int k = 0; k <= top; k++)
			if (dense[k] != 0)
				taps.push_back({k, dense[k]});
		filters.push_back(std::move(taps));
	}
	prevLog.assign(filters.size(), 0.0);
	out.assign(fe.diff ? 2 * filters.size() : filters.size(), 0.0);
}

bool Features::push(double x) {
	ring[pos] = x;
	if (++pos == win)
		pos = 0;
	if (++fill < hop)
		return false;
	fill = 0;

	for (int i = 0; i < win; i++)
		frame[i] = ring[(pos + i) % win] * window[i];
	std::fill(frame.begin() + win, frame.end(), 0.0);
	fft.forward(frame.data(), spec.data());
	for (size_t k = 0; k < mag.size(); k++)
		mag[k] = std::abs(spec[k]);

	int nb = int(filters.size());
	for (int j = 0; j < nb; j++) {
		double s = 0;
		for (const Tap& t : filters[j])
			s += mag[t.bin] * t.weight;
		double v = std::log10(1 + fe.logMul * s);
		out[j] = v;
		if (fe.diff)
			out[nb + j] = std::max(v - prevLog[j], 0.0);
		prevLog[j] = v;
	}
	return true;
}

} // namespace beat

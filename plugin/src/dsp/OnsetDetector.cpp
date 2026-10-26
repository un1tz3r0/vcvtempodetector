#include "OnsetDetector.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>

namespace beat {

static const double BAND_EDGES[2] = {200.0, 2000.0};

static int bandOf(double freq) {
	return (freq >= BAND_EDGES[0]) + (freq >= BAND_EDGES[1]);
}

OnsetDetector::OnsetDetector(double sampleRate, Odf kind, double fps)
	: sr(sampleRate),
	  kind(kind),
	  win(1 << int(std::ceil(std::log2(0.0464 * sampleRate)))),
	  hop_(int(std::nearbyint(sampleRate / fps))),
	  nbins(win / 2 + 1),
	  fft(win),
	  ring(win),
	  frame(win),
	  hann(win),
	  binBand(nbins),
	  spec(nbins),
	  prev1(nbins),
	  prev2(nbins),
	  mag(nbins) {
	for (int i = 0; i < win; i++)
		hann[i] = 0.5 - 0.5 * std::cos(2 * M_PI * i / win);
	for (int k = 0; k < nbins; k++)
		binBand[k] = bandOf(k * sr / win);

	// madmom's log filterbank: 24 bands per octave, 30 Hz to 17 kHz, area-normalised triangles.
	// Centre bins round half to even, as numpy does.
	std::fesetround(FE_TONEAREST);
	const double bpo = 24, fmin = 30, fmax = std::min(17000.0, sr / 2);
	std::vector<int> centres;
	for (double k = std::floor(bpo * std::log2(fmin / 440)); k <= std::ceil(bpo * std::log2(fmax / 440)); k++)
		centres.push_back(std::clamp(int(std::nearbyint(440 * std::pow(2.0, k / bpo) * win / sr)), 1, nbins - 1));
	centres.erase(std::unique(centres.begin(), centres.end()), centres.end());
	for (size_t j = 0; j + 2 < centres.size(); j++) {
		int a = centres[j], c = centres[j + 1], b = centres[j + 2];
		Filter f{a, std::vector<double>(b - a + 1), 0};
		for (int i = a; i <= c; i++)
			f.weights[i - a] = double(i - a) / (c - a);
		for (int i = c; i <= b; i++)
			f.weights[i - a] = 1.0 - double(i - c) / (b - c);
		double sum = 0, centroid = 0;
		for (double w : f.weights)
			sum += w;
		for (int i = 0; i <= b - a; i++) {
			f.weights[i] /= sum;
			centroid += f.weights[i] * (a + i) * sr / win;
		}
		f.band = bandOf(centroid);
		filters.push_back(std::move(f));
	}
	logspec.assign(filters.size(), 0.0);
	prevLog.assign(filters.size(), 0.0);
}

bool OnsetDetector::push(double x, OdfFrame& out) {
	ring[pos] = x;
	if (++pos == win)
		pos = 0;
	if (++fill < hop_)
		return false;
	fill = 0;
	analyse(out);
	return true;
}

void OnsetDetector::analyse(OdfFrame& out) {
	std::copy(ring.begin() + pos, ring.end(), frame.begin());
	std::copy(ring.begin(), ring.begin() + pos, frame.end() - pos);

	double energy = 0;
	for (int i = win - hop_; i < win; i++)
		energy += frame[i] * frame[i];
	out.env = std::sqrt(energy / hop_);

	for (int i = 0; i < win; i++)
		frame[i] *= hann[i];
	fft.forward(frame.data(), spec.data());

	std::fill(out.bands, out.bands + 3, 0.0);
	for (int k = 0; k < nbins; k++) {
		double power = spec[k].real() * spec[k].real() + spec[k].imag() * spec[k].imag();
		mag[k] = std::sqrt(power);
		out.bands[binBand[k]] += power;
	}

	std::fill(out.odf, out.odf + 3, 0.0);
	out.columns = columns();
	switch (kind) {
		case Odf::SuperFlux:
		case Odf::Flux: {
			int nf = filters.size();
			for (int j = 0; j < nf; j++) {
				const Filter& f = filters[j];
				double s = 0;
				for (size_t i = 0; i < f.weights.size(); i++)
					s += mag[f.start + i] * f.weights[i];
				logspec[j] = std::log10(1 + s);
			}
			for (int j = 0; j < nf; j++) {
				double ref = prevLog[j];
				if (kind == Odf::SuperFlux) {  // maximum filter across neighbouring bands suppresses vibrato
					if (j > 0)
						ref = std::max(ref, prevLog[j - 1]);
					if (j + 1 < nf)
						ref = std::max(ref, prevLog[j + 1]);
				}
				out.odf[filters[j].band] += std::max(logspec[j] - ref, 0.0);
			}
			std::swap(logspec, prevLog);
			break;
		}
		case Odf::Complex: {  // rectified complex domain (Dixon 2006)
			double sum = 0;
			for (int k = 0; k < nbins; k++) {
				double m1 = std::abs(prev1[k]);
				if (mag[k] > m1)
					sum += std::abs(spec[k] - std::polar(m1, 2 * std::arg(prev1[k]) - std::arg(prev2[k])));
			}
			out.odf[0] = sum;
			break;
		}
		case Odf::Hfc: {  // rectified rise of high frequency content
			double hfc = 0;
			for (int k = 0; k < nbins; k++)
				hfc += mag[k] * mag[k] * k;
			out.odf[0] = std::max(hfc - prevHfc, 0.0);
			prevHfc = hfc;
			break;
		}
	}
	std::swap(prev2, prev1);
	prev1 = spec;
}

} // namespace beat

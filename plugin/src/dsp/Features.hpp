#pragma once
#include <array>
#include <complex>
#include <vector>

#include "RealFft.hpp"

namespace beat {

/** A neural model's input features, field for field python/beatref/features.py Frontend. */
struct Frontend {
	int bandsPerOctave = 12;
	double fmin = 30, fmax = 17000, logMul = 1e4;
	bool diff = true;

	/** Filters as (left, centre, right) bins of madmom's filterbank at 44.1 kHz / 2048 points. */
	std::vector<std::array<int, 3>> triangles() const;
	int bands() const { return int(triangles().size()); }
	int size() const { return bands() * (diff ? 2 : 1); }
};

/** Causal, sample-by-sample port of python/beatref/features.py: frame i covers the 46.4 ms ending
at sample (i + 1) * hop. Per frame: log10(1 + mul * band magnitude) per band, then (if `diff`) the
positive difference to the previous frame. */
class Features {
public:
	Features(double sampleRate, const Frontend& fe, double fps);

	int size() const { return int(out.size()); }
	int hopSize() const { return hop; }

	/** Push one sample; returns true when it completes a frame, whose features are values(). */
	bool push(double x);
	const double* values() const { return out.data(); }

private:
	struct Tap {
		int bin;
		double weight;
	};

	Frontend fe;
	int win, nfft, hop, pos = 0, fill = 0;
	RealFft fft;
	std::vector<double> ring, frame, window, mag, prevLog, out;
	std::vector<std::complex<double>> spec;
	std::vector<std::vector<Tap>> filters;
};

} // namespace beat

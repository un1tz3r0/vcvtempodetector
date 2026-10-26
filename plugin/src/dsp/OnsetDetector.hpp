#pragma once
#include <complex>
#include <vector>

#include "Config.hpp"
#include "Fft.hpp"

namespace beat {

/** One analysis frame. The filterbank ODFs (SuperFlux, flux) give one value per band
(low < 200 Hz, mid, high > 2 kHz); complex domain and HFC give one. */
struct OdfFrame {
	double odf[3] = {};
	int columns = 1;
	double env = 0;          // RMS of the hop's samples
	double bands[3] = {};    // low / mid / high spectral power, for display
};

/** Causal, sample-by-sample port of python/beatref/odf.py: frame i covers the `win` samples
ending at sample (i + 1) * hop, with zeros before the first sample. */
class OnsetDetector {
public:
	OnsetDetector(double sampleRate, Odf kind, double fps);

	int hop() const { return hop_; }
	int window() const { return win; }
	double fps() const { return sr / hop_; }
	int columns() const { return kind == Odf::SuperFlux || kind == Odf::Flux ? 3 : 1; }

	/** Push one sample; returns true (and fills `out`) when it completes a frame. */
	bool push(double x, OdfFrame& out);

private:
	struct Filter {
		int start;
		std::vector<double> weights;
		int band;
	};

	void analyse(OdfFrame& out);

	double sr;
	Odf kind;
	int win, hop_, nbins;
	RealFft fft;
	std::vector<double> ring, frame, hann;
	std::vector<int> binBand;
	std::vector<Filter> filters;
	std::vector<std::complex<double>> spec, prev1, prev2;
	std::vector<double> mag, logspec, prevLog;
	double prevHfc = 0;
	int pos = 0, fill = 0;
};

} // namespace beat

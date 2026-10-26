#pragma once

// Every tunable of the tracker, field for field the same as python/beatref/config.py.

namespace beat {

enum class Odf { SuperFlux, Flux, Complex, Hfc };
enum class Lost { Coast, Stop, Follow };

struct Config {
	// tempo
	double minBpm = 60.0;
	double maxBpm = 200.0;
	double prefMinBpm = 90.0;
	double prefMaxBpm = 180.0;
	double tempoPrior = 0.002;       // per-frame log penalty per octave outside the preferred range

	// onset detection
	Odf odf = Odf::SuperFlux;
	double fps = 100.0;
	double latency = 0.013;          // ODF peak delay behind the true onset [s]

	// activation normalisation
	double normMeanTau = 1.0;
	double normPeakTau = 3.0;
	double normFloor = 0.1;
	double lowWeight = 2.0;
	double actGamma = 0.75;
	double actFloor = 0.01;
	double silenceDb = -60.0;

	// HMM
	double transitionLambda = 100.0;
	double observationLambda = 16.0;
	double restart = 0.0;

	// confidence and lock
	double confTau = 2.0;
	double confScale = 2.0;
	double lockOn = 0.6;
	double lockOff = 0.4;

	// oscillator
	double follow = 0.5;             // 0 rigid (2 s), 0.5 (0.2 s), 1 loose (0.02 s)
	double glideBeats = 1.0;
	Lost lost = Lost::Coast;
	int stopBars = 4;
	double phaseOffset = 0.0;        // degrees of a beat; positive fires earlier
	double timeOffset = 0.0;         // seconds; positive fires earlier
	int beatsPerBar = 4;
	double ratio = 2.0;              // Clock x/÷ output, in clocks per beat

	// seeding
	double warmBpm = 0.0;            // 0 = cold start
	double warmPhase = 0.0;
	double clockPull = 0.5;          // 1 = the clock takes over and audio is ignored
	double tapPull = 0.9;

	// module additions (not in the reference)
	bool seedFromClock = false;      // warm start: the first two Clock beats after a reset seed tempo and phase
	bool autoDownbeat = false;       // experimental: bar 1 is the beat with the most low-band onset energy
	double nudgeRate = 0.25;         // beats per second while a Nudge button is held

	/** Settings whose change rebuilds the tracker (and so restarts acquisition). */
	bool sameTracker(const Config& o) const {
		return minBpm == o.minBpm && maxBpm == o.maxBpm && prefMinBpm == o.prefMinBpm && prefMaxBpm == o.prefMaxBpm
			&& tempoPrior == o.tempoPrior && fps == o.fps && transitionLambda == o.transitionLambda
			&& observationLambda == o.observationLambda && restart == o.restart;
	}
};

} // namespace beat

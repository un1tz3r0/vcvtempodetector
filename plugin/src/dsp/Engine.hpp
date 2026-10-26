#pragma once
#include <array>
#include <cstdint>
#include <memory>

#include "BeatTracker.hpp"
#include "Config.hpp"
#include "OnsetDetector.hpp"
#include "Oscillator.hpp"

namespace beat {

/** Per-frame summary: what the display draws and what the comparison harness prints. */
struct Frame {
	double time = 0;            // end of the frame [s]
	float env = 0, bands[3] = {}, odf[3] = {}, act = 0;
	float bpm = 0, phase = 0, conf = 0;
	bool locked = false, held = false;
	float beatAt = -1;          // seconds before `time` of a beat fired during the frame, or -1
	bool bar = false;           // that beat was bar 1
	double outPhase = 0, bps = 0;  // output phase at `time` and its rate, to extrapolate future beats
	bool running = false;
	double barOrigin = 0;
	int beatsPerBar = 4;
};

/** Per-sample outputs. */
struct Output {
	bool beat = false, bar = false, clock = false;
	double beatPhase = 0, barPhase = 0;  // [0, 1)
};

/** The whole chain, sample by sample: onset detection, activation normalisation, tracker,
confidence and lock (python/beatref/detector.py), then the oscillator at audio rate with the
bar counter, the clock ×/÷ divider and the module's manual controls. */
class Engine {
public:
	Engine(const Config& cfg, double sampleRate);

	/** Apply new settings. Tempo range and HMM changes rebuild the tracker; an ODF change rebuilds
	the onset detector. Everything else applies on the next frame. */
	void configure(const Config& cfg);
	const Config& config() const { return cfg; }

	/** One mono sample, full scale ±1. */
	Output process(double x);

	bool frameReady() const { return newFrame; }
	const Frame& frame() const { return current; }

	// controls, applied at the current sample
	void tap();
	void clock();
	void reset();
	void barReset();
	void scaleTempo(double factor);
	void setHold(bool on) { held = on; }
	void setNudge(int direction) { nudgeDir = direction; }
	void setClockConnected(bool on) { clockConnected = on; }

	double bpm() const { return osc.running ? osc.bps * 60 : estimate.bpm; }
	double confidence() const { return conf; }
	bool locked() const { return lock; }
	bool running() const { return osc.running; }
	double fps() const { return detector->fps(); }
	int64_t samples() const { return sampleCount; }

private:
	void analyse(const OdfFrame& f);
	void schedule(double pull);
	void seed(double bpm, int64_t beatSample);
	void trackTaps(std::array<int64_t, 4>& times, int& count, int64_t at, bool seedNow);
	void resyncClock(double ref);
	void guessDownbeat(double lowAct);
	double barPos(double value) const;

	Config cfg;
	double sr, dt = 0.01;
	std::unique_ptr<OnsetDetector> detector;
	std::unique_ptr<BeatTracker> tracker;
	Oscillator osc;
	BeatTracker::Estimate estimate{120, 0, 0};

	// activation normalisation
	double mean[3] = {}, peak[3] = {}, longPeak[3] = {}, weights[3] = {};
	double kMean = 0, rPeak = 0, rLong = 0, kConf = 0, quiet = 0;

	// confidence
	double evidence = 0, conf = 0;
	bool lock = false;

	// timing
	int64_t sampleCount = 0, frames = 0;
	int segPos = 0;
	double value = 0;

	// events: (frame index, pull), applied when that frame is analysed
	std::array<std::pair<int64_t, double>, 16> events{};
	std::array<int64_t, 4> tapTimes{}, clockTimes{};
	int taps = 0, clocks = 0;
	bool clockSeeded = false;

	// bars and clock ×/÷
	double barOrigin = 0;
	bool barKnown = false;
	int64_t clockFired = 0;
	double lastRatio = 0;
	std::array<double, 16> downbeat{};

	bool held = false, clockConnected = false;
	int nudgeDir = 0, lastNudge = 0;

	Frame current;
	bool newFrame = false;
	int64_t pendingBeatAt = -1;
	bool pendingBar = false;
};

} // namespace beat

#pragma once
#include <cmath>
#include <cstdint>

#include "Config.hpp"

namespace beat {

/** Flywheel oscillator, ported from python/beatref/oscillator.py. Phase is unwrapped (in beats),
so beat n fires exactly once when the output phase first reaches n. The tracker steers it once
per frame (step); between frames the output phase moves linearly across the frame (render), so
beats land on the sample where they fall. */
class Oscillator {
public:
	bool running = false;
	double phase = 0;        // unwrapped beats
	double bps = 0;          // beats per second
	int64_t fired = 0;       // last beat index emitted

	void configure(const Config& c, double dt);

	/** Output phase: the unwrapped phase with both offsets applied. */
	double out() const { return phase + offset + timeOffset * bps; }

	/** Advance one frame, steered by the tracker's (bpm, beat phase at the end of the frame).
	Afterwards the output phase moves from `from` to `to` over the next frame, if `active`. */
	void step(double bpm, double target, bool locked, bool hold);

	/** Output phase at fraction `frac` (0, 1] of the current frame; true if it starts a new beat. */
	bool render(double frac, double& value) {
		value = from + (to - from) * frac;
		int64_t n = int64_t(std::floor(value));
		if (!active || n <= fired)
			return false;
		fired = n;
		return true;
	}

	/** ×2 / ÷2: rescale the beat grid around now, keeping existing beats on the new grid. */
	void scale(double f) {
		phase *= f;
		bps *= f;
		fired = int64_t(std::floor(out()));
		active = false;
	}

	/** Jump to tempo `newBps` with unwrapped phase `target` (mod 1) at the end of the current
	frame, starting if stopped. Used by Tap and Clock seeding. */
	void seed(double newBps, double target) {
		double err = target - phase + 0.5;
		err = err - std::floor(err) - 0.5;
		if (!running) {
			phase = std::floor(phase) + 1 + (target - std::floor(target));
			running = true;
			unlockedFor = 0;
		}
		else
			phase += err;
		bps = newBps;
		gliding = false;
		fired = int64_t(std::floor(out()));
		active = false;
	}

	bool active = false;
	double from = 0, to = 0;

private:
	Config cfg;
	double dt = 0.01, kPhase = 0, kTempo = 0, offset = 0, timeOffset = 0;
	bool gliding = false;
	double unlockedFor = 0;  // beats since lock was lost
};

} // namespace beat

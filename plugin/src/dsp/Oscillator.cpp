#include "Oscillator.hpp"

#include <algorithm>

namespace beat {

void Oscillator::configure(const Config& c, double frameDt) {
	cfg = c;
	dt = frameDt;
	double tau = 2.0 * std::pow(0.02 / 2.0, c.follow);
	kPhase = dt / tau;
	kTempo = dt / (2 * tau);
	offset = c.phaseOffset / 360;
	timeOffset = c.timeOffset;
}

void Oscillator::step(double bpm, double target, bool locked, bool hold) {
	active = false;
	if (!running) {
		if (locked && !hold) {  // first lock snaps: nothing downstream depends on the phase yet
			bps = bpm / 60;
			phase = std::floor(phase) + 1 + target;
			fired = int64_t(std::floor(out()));
			running = true;
			gliding = false;
			unlockedFor = 0;
		}
		return;
	}

	from = out();
	phase += bps * dt;
	to = out();
	active = true;
	if (hold)
		return;

	if (locked) {
		if (unlockedFor > 0)
			gliding = true;
		unlockedFor = 0;
	}
	else {
		unlockedFor += bps * dt;
		if (cfg.lost == Lost::Stop && unlockedFor >= cfg.stopBars * cfg.beatsPerBar)
			running = false;
		if (cfg.lost != Lost::Follow)
			return;
	}

	double err = target - phase + 0.5;
	err = err - std::floor(err) - 0.5;
	double kp = kPhase, kt = kTempo;
	if (gliding) {
		kp = kt = std::max(kPhase, dt * bps / cfg.glideBeats);
		gliding = std::abs(err) > 0.02;
	}
	phase += kp * err;
	bps += kt * (bpm / 60 - bps);
}

} // namespace beat

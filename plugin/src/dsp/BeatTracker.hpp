#pragma once
#include <vector>

#include "Config.hpp"

namespace beat {

/** Online beat tracker: madmom's DBNBeatTrackingProcessor in forward mode, as ported in
python/beatref/tracker.py. One chain of `interval` states per tempo, advancing one state per
frame; tempo changes only when a chain wraps. The first 1/observation_lambda of each beat is
the beat region. */
class BeatTracker {
public:
	struct Estimate {
		double bpm, phase, evidence;
	};

	BeatTracker(const Config& cfg, double fps);

	/** Uniform belief, or one concentrated near (bpm, phase) for a warm start. */
	void reset(double bpm = 0, double phase = 0);

	/** Rule out tempos more than 0.35 octave from `bpm` (0 lifts it), so a metrical level chosen
	with Tap, ×2 or ÷2 sticks: the tracker can still follow drift and ramps of about ±25%, but
	can't walk back to double or half tempo. */
	void pin(double bpm);

	/** Advance one frame. `act` < 0 means no audio evidence (Hold, or a clock at full pull);
	`event` > 0 is the pull of a Tap or Clock beat. Evidence is the log likelihood ratio of this
	frame under the prediction against a beat-agnostic null model. */
	Estimate step(double act, double event = 0);

	/** (bpm, phase) of the current belief. */
	Estimate estimate() const;

	/** Rotate every tempo chain by `beats` of a beat, accumulated across calls so slow nudges
	still move short chains; endNudge() forgets the accumulated fraction. */
	void nudge(double beats);
	void endNudge();

	double minBpm() const { return 60 * fps / intervals.back(); }
	double maxBpm() const { return 60 * fps / intervals.front(); }

private:
	double fps, lambda, restart;
	std::vector<int> intervals, first, last, tempoOf, applied;
	std::vector<double> pos, cosPos, sinPos, basePrior, prior, trans, p, q;
	mutable std::vector<double> chainMass;
	std::vector<char> isBeat;
	double nudged = 0;
};

} // namespace beat

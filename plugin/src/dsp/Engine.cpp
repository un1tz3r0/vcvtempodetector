#include "Engine.hpp"

#include <algorithm>
#include <cmath>

namespace beat {

static double wrap(double x, double m = 1) {
	return x - m * std::floor(x / m);
}

Engine::Engine(const Config& c, double sampleRate) : sr(sampleRate) {
	configure(c);
	reset();
}

void Engine::configure(const Config& c) {
	bool newDetector = !detector || c.odf != cfg.odf || c.fps != cfg.fps;
	bool newTracker = !tracker || newDetector || !c.sameTracker(cfg);
	cfg = c;
	if (newDetector) {
		detector.reset(new OnsetDetector(sr, cfg.odf, cfg.fps));
		std::fill(mean, mean + 3, 0.0);
		std::fill(peak, peak + 3, 0.0);
		std::fill(longPeak, longPeak + 3, 0.0);
		segPos = 0;
	}
	dt = 1 / detector->fps();
	if (newTracker && tracker)
		reset();
	if (newTracker)
		tracker.reset(new BeatTracker(cfg, detector->fps()));

	double w[3] = {cfg.lowWeight, 1, 1}, sum = 0;
	int cols = detector->columns();
	for (int i = 0; i < 3; i++)
		sum += weights[i] = i < cols ? w[i] : 0;
	for (double& v : weights)
		v /= sum;
	kMean = dt / cfg.normMeanTau;
	rPeak = std::exp(-dt / cfg.normPeakTau);
	rLong = std::exp(-dt / 20);
	kConf = dt / cfg.confTau;
	quiet = std::pow(10, cfg.silenceDb / 20);
	osc.configure(cfg, dt);
}

void Engine::reset() {
	if (tracker) {
		tracker->reset(cfg.warmBpm, cfg.warmPhase);
		tracker->pin(0);
	}
	evidence = -2 * cfg.confScale;  // start unconfident: lock must be earned
	conf = 1 / (1 + std::exp(-evidence / cfg.confScale));
	lock = false;
	taps = clocks = 0;
	clockSeeded = false;
	events.fill({-1, 0});
}

Output Engine::process(double x) {
	Output o;
	newFrame = false;
	if (osc.active) {
		if (osc.render(double(++segPos) / detector->hop(), value)) {
			o.beat = true;
			double n = osc.fired;
			if (!barKnown) {
				barOrigin = n;
				barKnown = true;
				resyncClock(n - 1e-6);
			}
			o.bar = barPos(n) < 0.5;
			pendingBeatAt = sampleCount;
			pendingBar = o.bar;
		}
		if (cfg.ratio != lastRatio)
			resyncClock(value);
		int64_t c = int64_t(std::floor((value - barOrigin) * cfg.ratio));
		if (c > clockFired) {
			clockFired = c;
			o.clock = true;
		}
	}
	o.beatPhase = value - std::floor(value);
	o.barPhase = barPos(value) / cfg.beatsPerBar;

	OdfFrame f;
	bool done = detector->push(x, f);
	if (neural) {
		if (fed)
			neural->push(x, frames, done && f.env > quiet);  // completes a frame exactly when the detector does
		fed = fed || done;
	}
	if (done)
		analyse(f);
	sampleCount++;
	return o;
}

std::unique_ptr<Inference> Engine::setInference(std::unique_ptr<Inference> next) {
	std::swap(neural, next);
	fed = sampleCount == 0;
	tracked = frames;
	downbeat.fill(0);
	return next;
}

void Engine::analyse(const OdfFrame& f) {
	double a = 0, low = 0;
	for (int c = 0; c < f.columns; c++) {
		double v = f.odf[c];
		mean[c] += kMean * (v - mean[c]);
		longPeak[c] = std::max(v, longPeak[c] * rLong);
		peak[c] = std::max({v, peak[c] * rPeak, cfg.normFloor * longPeak[c]});
		double n = std::clamp((v - mean[c]) / (peak[c] - mean[c] + 1e-12), 0.0, 1.0);
		a += n * weights[c];
		if (c == 0)
			low = n;
	}
	double act = std::clamp(f.env > quiet ? std::pow(a, cfg.actGamma) : 0.0, cfg.actFloor, 1 - cfg.actFloor);

	if (nudgeDir) {
		double d = nudgeDir * cfg.nudgeRate * dt;
		tracker->nudge(d);
		osc.phase += d;
	}
	else if (lastNudge)
		tracker->endNudge();
	lastNudge = nudgeDir;

	pendingDowns = 0;
	if (neural) {
		Inference::Result r;
		while (neural->pop(r)) {
			while (tracked < r.frame)
				track(-1, -1);  // a frame the worker dropped: the tracker predicts without observing
			track(std::clamp(r.loud ? r.beat : 0.0, cfg.actFloor, 1 - cfg.actFloor), neural->network().hasDownbeat() ? r.downbeat : -1);
		}
	}
	else
		track(act, f.columns == 3 ? low : -1);

	// the tracker's phase describes the audio `lag` frames plus the activation latency ago
	int64_t lag = frames + 1 - tracked;
	double bpm = estimate.bpm, phase = wrap(estimate.phase + (latency() + lag * dt) * bpm / 60);

	bool wasActive = osc.active;
	osc.step(bpm, phase + bpm / 60 * dt, lock, held);
	segPos = 0;
	if (osc.active && !wasActive)
		resyncClock(osc.from);
	for (int i = 0; i < pendingDowns; i++)
		guessDownbeat(pendingDown[i].second, frames - pendingDown[i].first);
	frames++;

	Frame& r = current;
	r.time = double(sampleCount + 1) / sr;
	r.env = f.env;
	for (int i = 0; i < 3; i++) {
		r.bands[i] = f.bands[i];
		r.odf[i] = f.odf[i];
	}
	r.act = shownAct;
	r.bpm = bpm;
	r.phase = phase;
	r.conf = conf;
	r.locked = lock;
	r.held = held;
	r.beatAt = pendingBeatAt >= 0 ? float((sampleCount + 1 - pendingBeatAt) / sr) : -1.f;
	r.bar = pendingBar;
	r.running = osc.running;
	r.outPhase = osc.active ? osc.from : value;
	r.bps = osc.bps;
	r.barOrigin = barOrigin;
	r.beatsPerBar = cfg.beatsPerBar;
	pendingBeatAt = -1;
	pendingBar = false;
	newFrame = true;
}

void Engine::track(double act, double down) {
	double pull = 0;
	for (auto& e : events)
		if (e.first >= 0 && e.first <= tracked) {
			if (e.first == tracked)
				pull = std::max(pull, e.second);
			e = {-1, 0};
		}
	bool deaf = clockConnected && cfg.clockPull >= 1;
	estimate = tracker->step(held || deaf || act < 0 ? -1 : act, pull);
	evidence += kConf * (estimate.evidence / dt - evidence);  // nats per second
	conf = 1 / (1 + std::exp(-evidence / cfg.confScale));
	lock = conf > (lock ? cfg.lockOff : cfg.lockOn);
	if (cfg.autoDownbeat && down >= 0 && pendingDowns < int(pendingDown.size()))
		pendingDown[pendingDowns++] = {tracked, down};
	if (act >= 0)
		shownAct = act;
	tracked++;
}

double Engine::barPos(double v) const {
	return wrap(v - barOrigin, cfg.beatsPerBar);
}

void Engine::resyncClock(double ref) {
	clockFired = int64_t(std::floor((ref - barOrigin) * cfg.ratio));
	lastRatio = cfg.ratio;
}

void Engine::schedule(double pull) {
	int64_t k = frames + int64_t(std::lround(latency() * detector->fps()));  // line up with the lagging activation
	auto slot = std::find_if(events.begin(), events.end(), [](auto& e) { return e.first < 0; });
	if (slot != events.end())
		*slot = {k, pull};
}

void Engine::trackTaps(std::array<int64_t, 4>& times, int& count, int64_t at, bool seedNow) {
	if (count && at - times[count - 1] > 2 * sr)
		count = 0;
	if (count == int(times.size())) {
		std::rotate(times.begin(), times.begin() + 1, times.end());
		count--;
	}
	times[count++] = at;
	if (count < 2 || !seedNow)
		return;
	double bpm = 60 * sr * (count - 1) / double(times[count - 1] - times[0]);
	if (bpm >= tracker->minBpm() * 0.97 && bpm <= tracker->maxBpm() * 1.03)
		seed(bpm, at);
}

void Engine::tap() {
	schedule(cfg.tapPull);
	trackTaps(tapTimes, taps, sampleCount, true);
}

void Engine::clock() {
	schedule(cfg.clockPull);
	trackTaps(clockTimes, clocks, sampleCount, cfg.seedFromClock && !clockSeeded && !lock);
}

void Engine::seed(double bpm, int64_t beatSample) {
	// The tracker's phase lags true time by the ODF latency; the oscillator's phase is true time
	// at the end of the frame now being rendered.
	double bps = bpm / 60, frameEnd = double(frames) * detector->hop(), trackedEnd = double(tracked) * detector->hop();
	tracker->reset(bpm, wrap((trackedEnd - beatSample) / sr * bps - latency() * bps));
	tracker->pin(bpm);
	osc.seed(bps, (frameEnd + detector->hop() - beatSample) / sr * bps);
	clockSeeded = true;
}

void Engine::scaleTempo(double f) {
	BeatTracker::Estimate e = tracker->estimate();
	double bpm = e.bpm * f;
	if (bpm < tracker->minBpm() * 0.97 || bpm > tracker->maxBpm() * 1.03)
		return;
	tracker->reset(bpm, wrap(e.phase * f));
	tracker->pin(bpm);
	if (osc.running) {
		osc.scale(f);
		barOrigin = std::floor(barOrigin * f);
		resyncClock(osc.out());
	}
}

void Engine::barReset() {
	barOrigin = std::round(value);
	barKnown = true;
	resyncClock(value);
	downbeat.fill(0);
}

void Engine::guessDownbeat(double act, int64_t lag) {
	int bpb = std::min<int>(cfg.beatsPerBar, downbeat.size());
	double decay = std::exp(-dt / 8.0);
	for (double& d : downbeat)
		d *= decay;
	if (!osc.running || !barKnown || bpb < 2)
		return;
	// the activation describes audio from (1 + lag frames + latency) before the oscillator's phase
	double u = osc.out() - ((1 + lag) * dt + latency()) * osc.bps, n = std::round(u);
	if (std::abs(u - n) < 0.08)
		downbeat[int(barPos(n))] += act;
	int best = std::max_element(downbeat.begin(), downbeat.begin() + bpb) - downbeat.begin();
	if (best && downbeat[best] > 1.25 * downbeat[0]) {
		barOrigin += best;
		std::rotate(downbeat.begin(), downbeat.begin() + best, downbeat.begin() + bpb);
		resyncClock(value);
	}
}

} // namespace beat

#include "BeatTracker.hpp"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>

namespace beat {

BeatTracker::BeatTracker(const Config& cfg, double fps)
	: fps(fps), lambda(cfg.observationLambda), restart(cfg.restart) {
	std::fesetround(FE_TONEAREST);
	for (int i = int(std::nearbyint(60 * fps / cfg.maxBpm)); i <= int(std::nearbyint(60 * fps / cfg.minBpm)); i++)
		intervals.push_back(i);
	int nt = intervals.size(), n = 0;
	for (int t = 0; t < nt; t++) {
		first.push_back(n);
		n += intervals[t];
		last.push_back(n - 1);
		for (int k = 0; k < intervals[t]; k++) {
			double ph = double(k) / intervals[t];
			tempoOf.push_back(t);
			pos.push_back(ph);
			cosPos.push_back(std::cos(2 * M_PI * ph));
			sinPos.push_back(std::sin(2 * M_PI * ph));
			isBeat.push_back(ph < 1 / lambda);
		}
	}

	trans.resize(nt * nt);
	for (int i = 0; i < nt; i++) {
		double sum = 0;
		for (int j = 0; j < nt; j++) {
			double t = std::exp(-cfg.transitionLambda * std::abs(double(intervals[j]) / intervals[i] - 1));
			trans[i * nt + j] = t <= DBL_EPSILON ? 0 : t;
			sum += trans[i * nt + j];
		}
		for (int j = 0; j < nt; j++)
			trans[i * nt + j] /= sum;
	}

	for (int s = 0; s < n; s++) {
		double bpm = 60 * fps / intervals[tempoOf[s]];
		double outside = std::max(std::log2(cfg.prefMinBpm / bpm), 0.0) + std::max(std::log2(bpm / cfg.prefMaxBpm), 0.0);
		basePrior.push_back(std::exp(-cfg.tempoPrior * outside));
	}
	prior = basePrior;
	p.resize(n);
	q.resize(n);
	chainMass.resize(nt);
	applied.assign(nt, 0);
	reset(cfg.warmBpm, cfg.warmPhase);
}

void BeatTracker::pin(double bpm) {
	for (size_t s = 0; s < prior.size(); s++)
		prior[s] = bpm > 0 && std::abs(std::log2(60 * fps / intervals[tempoOf[s]] / bpm)) > 0.35 ? 0 : basePrior[s];
}

void BeatTracker::reset(double bpm, double phase) {
	double sum = 0;
	for (size_t s = 0; s < p.size(); s++) {
		if (bpm > 0) {
			double octaves = std::log2(60 * fps / intervals[tempoOf[s]] / bpm) / 0.05;
			p[s] = std::exp(-0.5 * octaves * octaves + 8 * std::cos(2 * M_PI * (pos[s] - phase)));
		}
		else
			p[s] = 1;
		sum += p[s];
	}
	for (double& v : p)
		v /= sum;
}

BeatTracker::Estimate BeatTracker::step(double act, double event) {
	int n = p.size(), nt = intervals.size();
	std::copy(p.begin(), p.end() - 1, q.begin() + 1);
	for (int j = 0; j < nt; j++) {
		double s = 0;
		for (int i = 0; i < nt; i++)
			s += p[last[i]] * trans[i * nt + j];
		q[first[j]] = s;
	}
	if (restart > 0)
		for (double& v : q)
			v = (1 - restart) * v + restart / n;

	double inBeat = 0, total = 0;
	for (int s = 0; s < n; s++) {
		total += q[s];
		if (isBeat[s])
			inBeat += q[s];
	}
	double on = inBeat / total, evidence = 0;
	if (act >= 0) {
		double beat = act, rest = (1 - act) / (lambda - 1);
		for (int s = 0; s < n; s++)
			q[s] *= isBeat[s] ? beat : rest;
		evidence += std::log((on * beat + (1 - on) * rest) / (beat / lambda + (1 - 1 / lambda) * rest));
	}
	if (event > 0) {
		double w = 1 + (event < 1 ? event / (1 - event) : 1e6);
		for (int s = 0; s < n; s++)
			if (isBeat[s])
				q[s] *= w;
		evidence += std::log((on * w + 1 - on) / (w / lambda + 1 - 1 / lambda));
	}
	double sum = 0;
	for (int s = 0; s < n; s++)
		sum += q[s] *= prior[s];
	for (int s = 0; s < n; s++)
		p[s] = q[s] / sum;

	Estimate e = estimate();
	e.evidence = evidence;
	return e;
}

BeatTracker::Estimate BeatTracker::estimate() const {
	int nt = intervals.size();
	std::vector<double>& mass = chainMass;
	std::fill(mass.begin(), mass.end(), 0.0);
	double c = 0, s = 0;
	for (size_t i = 0; i < p.size(); i++) {
		mass[tempoOf[i]] += p[i];
		c += p[i] * cosPos[i];
		s += p[i] * sinPos[i];
	}
	int best = std::max_element(mass.begin(), mass.end()) - mass.begin();
	double num = 0, den = 0;
	for (int t = 0; t < nt; t++)
		if (std::abs(double(intervals[t]) / intervals[best] - 1) <= 0.03) {
			num += mass[t] * intervals[t];
			den += mass[t];
		}
	double phase = std::atan2(s, c) / (2 * M_PI);
	return {60 * fps * den / num, phase - std::floor(phase), 0};
}

void BeatTracker::nudge(double beats) {
	nudged += beats;
	for (size_t t = 0; t < intervals.size(); t++) {
		int len = intervals[t], shift = int(std::lround(nudged * len)), d = shift - applied[t];
		if (!d)
			continue;
		auto b = p.begin() + first[t];
		std::rotate(b, b + ((len - d % len) % len), b + len);
		applied[t] = shift;
	}
}

void BeatTracker::endNudge() {
	nudged = 0;
	std::fill(applied.begin(), applied.end(), 0);
}

} // namespace beat

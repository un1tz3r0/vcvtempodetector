// Offline driver for the module's DSP engine, for comparison against python/beatref.
// Reads raw little-endian float64 mono samples from stdin and prints "time<TAB>beat-in-bar" per
// beat, exactly what `beatref track` prints. Settings use the Python Config names.
//
//   beattrack --sr 44100 [--set name=value ...] [--frames frames.tsv] < audio.f64

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "../src/dsp/Engine.hpp"

using namespace beat;

static std::map<std::string, std::function<void(Config&, const std::string&)>> settings() {
	std::map<std::string, std::function<void(Config&, const std::string&)>> m;
	auto num = [&](const char* name, double Config::*field) {
		m[name] = [field](Config& c, const std::string& v) { c.*field = std::stod(v); };
	};
	auto integer = [&](const char* name, int Config::*field) {
		m[name] = [field](Config& c, const std::string& v) { c.*field = std::stoi(v); };
	};
	num("min_bpm", &Config::minBpm);
	num("max_bpm", &Config::maxBpm);
	num("pref_min_bpm", &Config::prefMinBpm);
	num("pref_max_bpm", &Config::prefMaxBpm);
	num("tempo_prior", &Config::tempoPrior);
	num("fps", &Config::fps);
	num("latency", &Config::latency);
	num("norm_mean_tau", &Config::normMeanTau);
	num("norm_peak_tau", &Config::normPeakTau);
	num("norm_floor", &Config::normFloor);
	num("low_weight", &Config::lowWeight);
	num("act_gamma", &Config::actGamma);
	num("act_floor", &Config::actFloor);
	num("silence_db", &Config::silenceDb);
	num("transition_lambda", &Config::transitionLambda);
	num("observation_lambda", &Config::observationLambda);
	num("restart", &Config::restart);
	num("conf_tau", &Config::confTau);
	num("conf_scale", &Config::confScale);
	num("lock_on", &Config::lockOn);
	num("lock_off", &Config::lockOff);
	num("follow", &Config::follow);
	num("glide_beats", &Config::glideBeats);
	integer("stop_bars", &Config::stopBars);
	num("phase_offset", &Config::phaseOffset);
	num("time_offset", &Config::timeOffset);
	integer("beats_per_bar", &Config::beatsPerBar);
	num("ratio", &Config::ratio);
	num("warm_bpm", &Config::warmBpm);
	num("warm_phase", &Config::warmPhase);
	num("clock_pull", &Config::clockPull);
	num("tap_pull", &Config::tapPull);
	m["odf"] = [](Config& c, const std::string& v) {
		static const std::map<std::string, Odf> kinds{
			{"superflux", Odf::SuperFlux}, {"flux", Odf::Flux}, {"complex", Odf::Complex}, {"hfc", Odf::Hfc}};
		c.odf = kinds.at(v);
	};
	m["lost"] = [](Config& c, const std::string& v) {
		static const std::map<std::string, Lost> modes{{"coast", Lost::Coast}, {"stop", Lost::Stop}, {"follow", Lost::Follow}};
		c.lost = modes.at(v);
	};
	return m;
}

int main(int argc, char** argv) {
	Config cfg;
	double sr = 44100;
	FILE* frames = nullptr;
	auto table = settings();
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		if (a == "--sr" && i + 1 < argc)
			sr = std::atof(argv[++i]);
		else if (a == "--frames" && i + 1 < argc)
			frames = std::fopen(argv[++i], "w");
		else if (a == "--set" && i + 1 < argc) {
			std::string kv = argv[++i];
			size_t eq = kv.find('=');
			auto it = table.find(kv.substr(0, eq));
			if (eq == std::string::npos || it == table.end()) {
				std::fprintf(stderr, "unknown setting %s\n", kv.c_str());
				return 2;
			}
			it->second(cfg, kv.substr(eq + 1));
		}
		else {
			std::fprintf(stderr, "usage: beattrack --sr RATE [--set name=value ...] [--frames FILE] < samples.f64\n");
			return 2;
		}
	}

	Engine engine(cfg, sr);
	std::vector<double> buf(1 << 16);
	int64_t n = 0, beats = 0;
	size_t got;
	while ((got = std::fread(buf.data(), sizeof(double), buf.size(), stdin)) > 0)
		for (size_t i = 0; i < got; i++, n++) {
			Output o = engine.process(buf[i]);
			if (o.beat)
				std::printf("%.6f\t%lld\n", double(n) / sr, (long long)(beats++ % cfg.beatsPerBar + 1));
			if (frames && engine.frameReady()) {
				const Frame& f = engine.frame();
				std::fprintf(frames, "%.6f\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n", f.time, f.odf[0], f.odf[1], f.odf[2],
					f.act, f.bpm, f.phase, f.conf, f.locked);
			}
		}
	if (frames)
		std::fclose(frames);
	return 0;
}

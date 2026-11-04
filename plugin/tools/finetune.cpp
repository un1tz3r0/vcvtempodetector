// Learning mode's fine-tuning, as the module runs it (src/dsp/Trainer), from the command line:
// the C++ counterpart of `beatref finetune`, and how it is checked against it.
//
//   finetune BASE.json OUT.json DIR_OR_STEM... [--set name=value ...]
//
// Each DIR contributes every <stem>.wav with a <stem>.beats beside it. Settings are Tune's fields
// (layers, epochs, steps, batch, crop, lr, l2sp, dropout, holdout, seed).

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/dsp/Trainer.hpp"

namespace fs = std::filesystem;
using namespace beat;

int main(int argc, char** argv) {
	std::vector<std::string> args(argv + 1, argv + argc), sources;
	Tune t;
	for (size_t i = 0; i < args.size(); i++) {
		if (args[i] != "--set") {
			sources.push_back(args[i]);
			continue;
		}
		std::string kv = args.at(++i), k = kv.substr(0, kv.find('=')), v = kv.substr(kv.find('=') + 1);
		if (k == "layers") t.layers = std::stoi(v);
		else if (k == "epochs") t.epochs = std::stoi(v);
		else if (k == "steps") t.steps = std::stoi(v);
		else if (k == "batch") t.batch = std::stoi(v);
		else if (k == "crop") t.crop = std::stoi(v);
		else if (k == "lr") t.lr = std::stod(v);
		else if (k == "l2sp") t.l2sp = std::stod(v);
		else if (k == "dropout") t.dropout = std::stod(v);
		else if (k == "holdout") t.holdout = std::stod(v);
		else if (k == "seed") t.seed = std::stoull(v);
		else {
			std::fprintf(stderr, "unknown setting %s\n", k.c_str());
			return 2;
		}
	}
	if (sources.size() < 3) {
		std::fprintf(stderr, "usage: finetune BASE.json OUT.json DIR_OR_STEM... [--set name=value ...]\n");
		return 2;
	}
	std::string base = sources[0], out = sources[1];
	std::vector<std::string> stems;
	for (size_t i = 2; i < sources.size(); i++) {
		if (fs::is_directory(sources[i])) {
			std::vector<std::string> found;
			for (const auto& e : fs::directory_iterator(sources[i]))
				if (e.path().extension() == ".wav" && fs::exists(fs::path(e.path()).replace_extension(".beats")))
					found.push_back(fs::path(e.path()).replace_extension().string());
			std::sort(found.begin(), found.end());
			stems.insert(stems.end(), found.begin(), found.end());
		}
		else
			stems.push_back(sources[i]);
	}
	try {
		std::ifstream in(base);
		std::stringstream text;
		text << in.rdbuf();
		Trainer trainer(text.str(), fs::path(base).filename().string(), t);
		std::vector<Recording> recs;
		for (const std::string& s : stems)
			recs.push_back(Recording::load(s));
		std::atomic<bool> cancel{false};
		std::string adapted = trainer.run(recs, cancel, [&](const Trainer::Progress& p) {
			if (p.stage == "Validated")
				std::printf("epoch %3d  val %.4f  (base %.4f, best %.4f)\n", p.epoch, p.lastLoss, p.baseLoss, p.bestLoss);
			else if (p.stage != "Adapting")
				std::printf("%s\n", p.stage.c_str());
			std::fflush(stdout);
		});
		if (adapted.empty())
			return 1;
		std::ofstream(out) << adapted;
	}
	catch (const std::exception& e) {
		std::fprintf(stderr, "%s\n", e.what());
		return 1;
	}
	return 0;
}

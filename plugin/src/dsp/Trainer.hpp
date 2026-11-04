#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace beat {

/** Learning mode's fine-tuning in C++: python/beatref/finetune.py for the models `beatref train`
exports (dense, causal convolutions, dense with sigmoid outputs). The frozen layers run once per
recording and stretch factor, so each step only runs the top `layers` layers forward and back.
Differences from the Python: one gain per stretch factor, crops start on a 2 s grid, layers = 0 adapts all
but the first layer, and the latency is kept rather than recalibrated. */
struct Tune {
	int layers = 3;           // top layers to adapt (output first)
	int epochs = 8;
	int steps = 100;
	int batch = 16;
	int crop = 1000;          // frames per training sequence
	double lr = 1e-3;
	double l2sp = 1e-4;       // pull towards the base weights
	double dropout = 0.3;     // above train.py's 0.1: the frozen layers here run without it
	double holdout = 0.2;     // tail of each recording held back for validation
	uint64_t seed = 0;
};

/** Audio and its beat grid: <stem>.wav (PCM or float, any channel count, mixed to mono) and
<stem>.beats (`time [beat-in-bar]` per line). */
struct Recording {
	std::string name;
	double sampleRate = 0;
	std::vector<float> audio;
	std::vector<double> beats;
	std::vector<int> numbers;  // empty if the grid has no bar positions
	static Recording load(const std::string& stem);  // throws std::runtime_error
};

class Trainer {
public:
	struct Progress {
		std::string stage;        // what it is doing, for the menu
		int epoch = 0, epochs = 0;
		double baseLoss = 0, bestLoss = 0, lastLoss = 0;
	};

	/** Throws std::runtime_error if the model isn't one it can adapt. */
	Trainer(const std::string& modelJson, const std::string& baseName, const Tune& tune = Tune());
	~Trainer();

	/** Fine-tune on the recordings. Returns the adapted model's JSON, or "" if no epoch beat the
	base model on the held-back audio or `cancel` was set. Calls `report` as it goes. */
	std::string run(const std::vector<Recording>& recordings, const std::atomic<bool>& cancel,
		const std::function<void(const Progress&)>& report = nullptr);

	const Progress& progress() const { return prog; }

private:
	struct Impl;
	Impl* impl;
	Progress prog;
};

} // namespace beat

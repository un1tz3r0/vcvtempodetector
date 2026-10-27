#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "Features.hpp"
#include "Network.hpp"

namespace beat {

/** Feeds a Network one frame at a time and returns its activations in order, tagged with their
frame index. Threaded, the features and the network run on their own worker behind lock-free
single-producer single-consumer rings, so the audio thread only copies samples and never waits;
otherwise they run inline, which is deterministic (the offline driver and the parity tests). Build
it off the audio thread. */
class Inference {
public:
	struct Result {
		int64_t frame;
		double beat, downbeat;
		bool loud;  // the frame was above the silence threshold
	};

	Inference(std::unique_ptr<Network> net, double sampleRate, double fps, bool threaded);
	~Inference();

	const Network& network() const { return *net; }

	/** One audio sample; returns true when it completes a frame and the frame was queued. */
	bool push(double x, int64_t frame, bool loud);
	/** The next activation, if one is ready. */
	bool pop(Result& r);

private:
	static constexpr int SLOTS = 64;  // 0.64 s of frames
	struct Job {
		int64_t frame;
		bool loud;
		std::vector<double> x;  // the frame's hop of samples
	};

	void run();
	void infer(const Job& j);

	std::unique_ptr<Network> net;
	Features features;
	int fill = 0;
	Job jobs[SLOTS];
	Result results[SLOTS];
	std::atomic<int> jobHead{0}, jobTail{0}, resHead{0}, resTail{0};
	bool threaded;
	std::atomic<bool> quit{false};
	std::thread worker;
};

} // namespace beat

#include "Inference.hpp"

#include <chrono>

namespace beat {

Inference::Inference(std::unique_ptr<Network> n, double sampleRate, double fps, bool threaded)
	: net(std::move(n)), features(sampleRate, net->frontend(), fps), threaded(threaded) {
	for (Job& j : jobs)
		j.x.resize(features.hopSize());
	if (threaded)
		worker = std::thread(&Inference::run, this);
}

Inference::~Inference() {
	quit = true;
	if (worker.joinable())
		worker.join();
}

bool Inference::push(double x, int64_t frame, bool loud) {
	// The head slot is never the worker's, even when the ring is full.
	int head = jobHead.load(std::memory_order_relaxed), next = (head + 1) % SLOTS;
	Job& j = jobs[head];
	j.x[fill] = x;
	if (++fill < int(j.x.size()))
		return false;
	fill = 0;
	j.frame = frame;
	j.loud = loud;
	if (!threaded) {
		infer(j);
		return true;
	}
	if (next == jobTail.load(std::memory_order_acquire))
		return false;  // the worker is more than SLOTS frames behind: drop this frame
	jobHead.store(next, std::memory_order_release);
	return true;
}

bool Inference::pop(Result& r) {
	int tail = resTail.load(std::memory_order_relaxed);
	if (tail == resHead.load(std::memory_order_acquire))
		return false;
	r = results[tail];
	resTail.store((tail + 1) % SLOTS, std::memory_order_release);
	return true;
}

void Inference::infer(const Job& j) {
	for (double x : j.x)
		features.push(x);  // completes a frame on the last sample
	Result r{j.frame, 0, 0, j.loud};
	net->forward(features.values(), r.beat, r.downbeat);
	int head = resHead.load(std::memory_order_relaxed), next = (head + 1) % SLOTS;
	if (next == resTail.load(std::memory_order_acquire))
		return;  // nobody is reading results: drop
	results[head] = r;
	resHead.store(next, std::memory_order_release);
}

void Inference::run() {
	while (!quit) {
		int tail = jobTail.load(std::memory_order_relaxed);
		if (tail == jobHead.load(std::memory_order_acquire)) {
			// Poll rather than be notified, so the audio thread never makes a system call.
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			continue;
		}
		infer(jobs[tail]);
		jobTail.store((tail + 1) % SLOTS, std::memory_order_release);
	}
}

} // namespace beat

#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace beat {

/** Learning mode's Record: writes mono audio to <stem>.wav (32-bit float) and the clock grid
heard alongside it to <stem>.beats, in the `time [beat-in-bar]` layout `beatref finetune` reads.
The audio thread only copies into lock-free single-producer single-consumer rings; a worker writes
the file. Bar positions are written only if a bar reset arrived while recording: the beat nearest
each one is bar 1, and the beats before the first are numbered back from it. Construct and destroy
off the audio thread; the destructor finishes both files. */
class Recorder {
public:
	Recorder(const std::string& stem, double sampleRate, int beatsPerBar);
	~Recorder();

	bool ok() const { return file != nullptr; }
	/** One sample, in the module's ±1 full scale. */
	void push(float x);
	/** A clock beat or a bar reset at the current sample. */
	void beat() { mark(false); }
	void barReset() { mark(true); }

	double seconds() const { return double(samples.load(std::memory_order_relaxed)) / sampleRate; }
	int beats() const { return beatCount.load(std::memory_order_relaxed); }
	/** Samples lost because the disk fell more than RING samples behind. */
	int64_t dropped() const { return lost.load(std::memory_order_relaxed); }
	const std::string& path() const { return stem; }

private:
	static constexpr int RING = 1 << 18, MARKS = 4096;
	struct Mark {
		int64_t sample;
		bool bar;
	};

	void mark(bool bar);
	void run();
	void drain();

	std::string stem;
	double sampleRate;
	int beatsPerBar;
	FILE* file = nullptr;
	std::vector<float> ring;
	Mark marks[MARKS];
	std::atomic<int> head{0}, tail{0}, markHead{0}, markTail{0};
	std::atomic<int64_t> samples{0}, lost{0};
	std::atomic<int> beatCount{0};
	std::vector<int64_t> beatSamples, barSamples;
	int64_t written = 0;
	std::atomic<bool> quit{false};
	std::thread worker;
};

} // namespace beat

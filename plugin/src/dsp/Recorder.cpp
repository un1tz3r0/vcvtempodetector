#include "Recorder.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace beat {

namespace {
void put32(FILE* f, uint32_t v) {
	uint8_t b[4] = {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)};
	std::fwrite(b, 1, 4, f);
}

void put16(FILE* f, uint16_t v) {
	uint8_t b[2] = {uint8_t(v), uint8_t(v >> 8)};
	std::fwrite(b, 1, 2, f);
}

/** A mono IEEE float WAV header for `n` samples. */
void header(FILE* f, uint32_t rate, uint32_t n) {
	std::fseek(f, 0, SEEK_SET);
	std::fwrite("RIFF", 1, 4, f);
	put32(f, 36 + 4 * n);
	std::fwrite("WAVEfmt ", 1, 8, f);
	put32(f, 16);
	put16(f, 3);  // IEEE float
	put16(f, 1);
	put32(f, rate);
	put32(f, 4 * rate);
	put16(f, 4);
	put16(f, 32);
	std::fwrite("data", 1, 4, f);
	put32(f, 4 * n);
}
} // namespace

Recorder::Recorder(const std::string& stem, double sampleRate, int beatsPerBar)
	: stem(stem), sampleRate(sampleRate), beatsPerBar(std::max(beatsPerBar, 1)), ring(RING) {
	file = std::fopen((stem + ".wav").c_str(), "wb");
	if (!file)
		return;
	header(file, uint32_t(sampleRate), 0);
	worker = std::thread(&Recorder::run, this);
}

void Recorder::push(float x) {
	int h = head.load(std::memory_order_relaxed), next = (h + 1) % RING;
	if (next == tail.load(std::memory_order_acquire)) {
		lost.fetch_add(1, std::memory_order_relaxed);
		return;  // the grid stays aligned with the audio that was kept
	}
	ring[h] = x;
	head.store(next, std::memory_order_release);
	samples.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::mark(bool bar) {
	int h = markHead.load(std::memory_order_relaxed), next = (h + 1) % MARKS;
	if (next == markTail.load(std::memory_order_acquire))
		return;
	marks[h] = {samples.load(std::memory_order_relaxed), bar};
	markHead.store(next, std::memory_order_release);
	if (!bar)
		beatCount.fetch_add(1, std::memory_order_relaxed);
}

void Recorder::drain() {
	int t = tail.load(std::memory_order_relaxed), h = head.load(std::memory_order_acquire);
	while (t != h) {
		int end = h > t ? h : RING;
		written += std::fwrite(&ring[t], sizeof(float), end - t, file);
		t = end % RING;
		tail.store(t, std::memory_order_release);
	}
	for (int m = markTail.load(std::memory_order_relaxed); m != markHead.load(std::memory_order_acquire); m = (m + 1) % MARKS) {
		(marks[m].bar ? barSamples : beatSamples).push_back(marks[m].sample);
		markTail.store((m + 1) % MARKS, std::memory_order_release);
	}
}

void Recorder::run() {
	while (!quit) {
		drain();
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	drain();
}

Recorder::~Recorder() {
	if (!file)
		return;
	quit = true;
	worker.join();
	header(file, uint32_t(sampleRate), uint32_t(written));
	std::fclose(file);

	// Bar 1 is the beat nearest each bar reset; positions count on from there, and back before the first.
	size_t n = beatSamples.size();
	std::vector<size_t> anchors;
	for (int64_t r : barSamples) {
		size_t i = std::lower_bound(beatSamples.begin(), beatSamples.end(), r) - beatSamples.begin();
		if (i > 0 && (i == n || r - beatSamples[i - 1] < beatSamples[i] - r))
			i--;
		if (i < n)
			anchors.push_back(i);
	}
	std::sort(anchors.begin(), anchors.end());
	FILE* f = std::fopen((stem + ".beats").c_str(), "w");
	if (!f)
		return;
	for (size_t j = 0; j < n; j++) {
		double t = double(beatSamples[j]) / sampleRate;
		if (anchors.empty()) {
			std::fprintf(f, "%.6f\n", t);
			continue;
		}
		auto a = std::upper_bound(anchors.begin(), anchors.end(), j);
		int k = int(j) - int(a == anchors.begin() ? *a : *(a - 1));
		std::fprintf(f, "%.6f\t%d\n", t, (k % beatsPerBar + beatsPerBar) % beatsPerBar + 1);
	}
	std::fclose(f);
}

} // namespace beat

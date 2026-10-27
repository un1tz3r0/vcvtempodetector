#pragma once
#include <memory>
#include <string>

#include "Features.hpp"

namespace RTNeural {
template <typename T>
class Model;
}

namespace beat {

/** A beat network loaded from an RTNeural JSON file (python/beatref/network.py is the reference).
Outputs: 1 = beat; 2 = beat and downbeat; 3 = softmax over (none, beat, downbeat). The `beatref`
block in the file gives the front end and the activation latency. Stateful: one per engine. */
class Network {
public:
	/** Parse a model; throws std::runtime_error with a readable message if it can't be used. */
	static std::unique_ptr<Network> fromJson(const std::string& text, const std::string& name);
	static std::unique_ptr<Network> fromFile(const std::string& path);
	~Network();

	const std::string& name() const { return name_; }
	const Frontend& frontend() const { return fe; }
	double latency() const { return latency_; }
	bool hasDownbeat() const { return outputs > 1; }

	void reset();
	/** One frame of features in; the beat and downbeat activations out (downbeat 0 if none). */
	void forward(const double* features, double& beat, double& downbeat);

private:
	Network() = default;
	std::unique_ptr<RTNeural::Model<double>> model;
	std::string name_;
	Frontend fe;
	double latency_ = 0.015;
	int outputs = 1;
};

} // namespace beat

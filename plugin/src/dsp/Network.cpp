#define RTNEURAL_DEFAULT_ALIGNMENT 16
#define RTNEURAL_NO_DEBUG 1
#include <RTNeural/config.h>
#include <RTNeural/model_loader.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "Network.hpp"

namespace beat {

std::unique_ptr<Network> Network::fromJson(const std::string& text, const std::string& name) {
	std::unique_ptr<Network> n(new Network);
	n->name_ = name;
	try {
		nlohmann::json spec = nlohmann::json::parse(text);
		nlohmann::json meta = spec.value("beatref", nlohmann::json::object());
		nlohmann::json fe = meta.value("frontend", nlohmann::json::object());
		n->fe.bandsPerOctave = fe.value("bands_per_octave", n->fe.bandsPerOctave);
		n->fe.fmin = fe.value("fmin", n->fe.fmin);
		n->fe.fmax = fe.value("fmax", n->fe.fmax);
		n->fe.logMul = fe.value("log_mul", n->fe.logMul);
		n->fe.diff = fe.value("diff", n->fe.diff);
		n->latency_ = meta.value("latency", n->latency_);
		n->model = RTNeural::json_parser::parseJson<double>(spec);
	}
	catch (const std::exception& e) {
		throw std::runtime_error(name + ": not a usable RTNeural model (" + e.what() + ")");
	}
	if (!n->model || n->model->layers.empty())
		throw std::runtime_error(name + ": no layers RTNeural could build");
	if (n->model->getInSize() != n->fe.size())
		throw std::runtime_error(name + ": takes " + std::to_string(n->model->getInSize()) + " inputs, but its front end makes "
			+ std::to_string(n->fe.size()));
	n->outputs = n->model->getOutSize();
	if (n->outputs < 1 || n->outputs > 3)
		throw std::runtime_error(name + ": has " + std::to_string(n->outputs) + " outputs; expected 1, 2 or 3");
	n->reset();
	return n;
}

std::unique_ptr<Network> Network::fromFile(const std::string& path) {
	std::ifstream in(path);
	if (!in)
		throw std::runtime_error("can't open " + path);
	std::stringstream text;
	text << in.rdbuf();
	return fromJson(text.str(), path.substr(path.find_last_of("/\\") + 1));
}

Network::~Network() = default;

void Network::reset() {
	model->reset();
}

void Network::forward(const double* features, double& beat, double& downbeat) {
	model->forward(features);
	const double* y = model->getOutputs();
	beat = outputs == 3 ? y[1] + y[2] : y[0];
	downbeat = outputs == 3 ? y[2] : outputs == 2 ? y[1] : 0;
}

} // namespace beat

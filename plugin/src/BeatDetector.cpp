#include "BeatDetector.hpp"

#include <osdialog.h>

#include <fstream>
#include <sstream>

#include "Waterfall.hpp"

const std::vector<float> BeatDetector::RATIOS = {1 / 8.f, 1 / 6.f, 1 / 4.f, 1 / 3.f, 1 / 2.f, 2 / 3.f, 1, 3 / 2.f, 2, 3, 4, 6, 8};
const std::vector<std::string> BeatDetector::RATIO_LABELS = {"÷8", "÷6", "÷4", "÷3", "÷2", "×2/3", "×1", "×3/2", "×2", "×3", "×4", "×6", "×8"};
const std::vector<float> BeatDetector::ZOOMS = {3, 6, 12, 24};

BeatDetector::BeatDetector() {
	config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);

	configParam(FOLLOW_PARAM, 0.f, 1.f, 0.5f, "Follow", "%", 0.f, 100.f)->description =
		"How fast tempo and phase corrections are absorbed: 0% rigid (2 s), 50% (0.2 s), 100% loose (0.02 s)";
	configParam(PHASE_OFFSET_PARAM, -180.f, 180.f, 0.f, "Phase offset", "°")->description = "Positive fires earlier";
	configParam(TIME_OFFSET_PARAM, -250.f, 250.f, 0.f, "Time offset", " ms")->description =
		"Positive fires earlier, to cancel delay downstream";
	configSwitch(RATIO_PARAM, 0.f, RATIOS.size() - 1, 8.f, "Clock ×/÷ ratio", RATIO_LABELS);
	configParam(BEATS_PER_BAR_PARAM, 1.f, 16.f, 4.f, "Beats per bar")->snapEnabled = true;
	configButton(TAP_PARAM, "Tap tempo (2+ taps seed the tempo, each tap pulls the phase)");
	configButton(HALF_PARAM, "Halve tempo (÷2)");
	configButton(DOUBLE_PARAM, "Double tempo (×2)");
	configButton(NUDGE_DOWN_PARAM, "Nudge later (hold)");
	configButton(NUDGE_UP_PARAM, "Nudge earlier (hold)");
	configButton(BAR_RESET_PARAM, "Bar reset (nearest beat becomes bar 1)");
	configButton(RESET_PARAM, "Reset tracking");

	configParam(MIN_BPM_PARAM, 30.f, 300.f, 60.f, "Slowest tempo", " BPM")->snapEnabled = true;
	configParam(MAX_BPM_PARAM, 30.f, 300.f, 200.f, "Fastest tempo", " BPM")->snapEnabled = true;
	configParam(PREF_MIN_BPM_PARAM, 30.f, 300.f, 90.f, "Preferred from", " BPM")->snapEnabled = true;
	configParam(PREF_MAX_BPM_PARAM, 30.f, 300.f, 180.f, "Preferred to", " BPM")->snapEnabled = true;
	configParam(WARM_BPM_PARAM, 30.f, 300.f, 120.f, "Warm start tempo", " BPM");
	configParam(LOCK_THRESHOLD_PARAM, 0.3f, 0.9f, 0.6f, "Lock threshold", "%", 0.f, 100.f)->description =
		"Lock drops 20% below this";
	configParam(GLIDE_PARAM, 0.25f, 8.f, 1.f, "Re-lock glide", " beats");
	configParam(STOP_BARS_PARAM, 1.f, 64.f, 4.f, "Stop after", " bars")->snapEnabled = true;
	configParam(CLOCK_PULL_PARAM, 0.f, 1.f, 0.5f, "Clock pull", "%", 0.f, 100.f)->description =
		"At 100% the clock takes over and the audio is ignored";
	configParam(TAP_PULL_PARAM, 0.f, 0.99f, 0.9f, "Tap pull", "%", 0.f, 100.f);
	configParam(NUDGE_RATE_PARAM, 0.02f, 1.f, 0.25f, "Nudge speed", " beats/s");
	configParam(TRIGGER_WIDTH_PARAM, 0.1f, 20.f, 1.f, "Trigger width", " ms");

	configParam(LATENCY_PARAM, 0.f, 50.f, 13.f, "Onset latency", " ms");
	configParam(LOW_WEIGHT_PARAM, 0.25f, 4.f, 2.f, "Low band weight", "×");
	configParam(GAMMA_PARAM, 0.25f, 1.5f, 0.75f, "Activation gamma");
	configParam(TRANSITION_LAMBDA_PARAM, 10.f, 300.f, 100.f, "Tempo stiffness (transition λ)");
	configParam(OBSERVATION_LAMBDA_PARAM, 4.f, 64.f, 16.f, "Beat region (observation λ)")->snapEnabled = true;
	configParam(TEMPO_PRIOR_PARAM, 0.f, 0.02f, 0.002f, "Preferred range strength");
	configParam(CONF_TAU_PARAM, 0.25f, 8.f, 2.f, "Confidence smoothing", " s");
	configParam(CONF_SCALE_PARAM, 0.5f, 8.f, 2.f, "Confidence scale", " nats/s");
	configParam(SILENCE_PARAM, -96.f, -20.f, -60.f, "Silence threshold", " dBFS");

	configInput(AUDIO_L_INPUT, "Audio L / mono");
	configInput(AUDIO_R_INPUT, "Audio R");
	configInput(CLOCK_INPUT, "External clock");
	configInput(TAP_INPUT, "Tap");
	configInput(RESET_INPUT, "Reset");
	configInput(BAR_RESET_INPUT, "Bar reset");
	configInput(HOLD_INPUT, "Hold (gate)");
	configOutput(BEAT_OUTPUT, "Beat trigger");
	configOutput(BEAT_PHASE_OUTPUT, "Beat phase (0–10 V saw)");
	configOutput(BAR_OUTPUT, "Bar trigger");
	configOutput(BAR_PHASE_OUTPUT, "Bar phase (0–10 V saw)");
	configOutput(CLOCK_OUTPUT, "Clock ×/÷");
	configOutput(BPM_OUTPUT, "BPM CV");
	configOutput(LOCK_OUTPUT, "Lock gate");
	configOutput(CONFIDENCE_OUTPUT, "Confidence (0–10 V)");
	configLight(BEAT_LIGHT, "Beat");
	configLight(BAR_LIGHT, "Bar");
	configLight(LOCK_LIGHT, "Lock");

	configDivider.setDivision(64);
	lightDivider.setDivision(32);
	engine.reset(new beat::Engine(settings(), APP->engine->getSampleRate()));
	applyModel();
}

BeatDetector::~BeatDetector() {
	delete incoming.exchange(nullptr);
	delete retired.exchange(nullptr);
}

static std::string readFile(const std::string& path) {
	std::ifstream in(path);
	std::stringstream text;
	text << in.rdbuf();
	return text.str();
}

bool BeatDetector::applyModel(double sampleRate) {
	collect();  // also frees the old model where no widget runs, as in headless Rack
	std::unique_ptr<Swap> swap(new Swap);
	if (neural) {
		static const std::string builtin = readFile(asset::plugin(pluginInstance, "res/models/beat.json"));
		try {
			bool own = !modelJson.empty();
			swap->next.reset(new beat::Inference(beat::Network::fromJson(own ? modelJson : builtin, own ? modelName : "built-in model"),
				sampleRate > 0 ? sampleRate : APP->engine->getSampleRate(), beat::Config().fps, true));
		}
		catch (const std::exception& e) {
			modelError = e.what();
			WARN("Beat Detector: %s", e.what());
			return false;
		}
	}
	modelError.clear();
	delete incoming.exchange(swap.release());  // supersedes a swap the audio thread hasn't taken yet
	return true;
}

void BeatDetector::collect() {
	delete retired.exchange(nullptr);
}

beat::Config BeatDetector::settings() {
	auto p = [&](int id) { return double(params[id].getValue()); };
	beat::Config c;
	c.minBpm = std::min(p(MIN_BPM_PARAM), p(MAX_BPM_PARAM));
	c.maxBpm = std::max(c.minBpm + 10, std::max(p(MIN_BPM_PARAM), p(MAX_BPM_PARAM)));
	c.prefMinBpm = std::min(p(PREF_MIN_BPM_PARAM), p(PREF_MAX_BPM_PARAM));
	c.prefMaxBpm = std::max(p(PREF_MIN_BPM_PARAM), p(PREF_MAX_BPM_PARAM));
	c.tempoPrior = p(TEMPO_PRIOR_PARAM);
	c.odf = beat::Odf(odf);
	c.latency = p(LATENCY_PARAM) / 1000;
	c.lowWeight = p(LOW_WEIGHT_PARAM);
	c.actGamma = p(GAMMA_PARAM);
	c.silenceDb = p(SILENCE_PARAM);
	c.transitionLambda = p(TRANSITION_LAMBDA_PARAM);
	c.observationLambda = p(OBSERVATION_LAMBDA_PARAM);
	c.confTau = p(CONF_TAU_PARAM);
	c.confScale = p(CONF_SCALE_PARAM);
	c.lockOn = p(LOCK_THRESHOLD_PARAM);
	c.lockOff = std::max(c.lockOn - 0.2, 0.05);
	c.follow = p(FOLLOW_PARAM);
	c.glideBeats = p(GLIDE_PARAM);
	c.lost = beat::Lost(lost);
	c.stopBars = int(p(STOP_BARS_PARAM));
	c.phaseOffset = p(PHASE_OFFSET_PARAM);
	c.timeOffset = p(TIME_OFFSET_PARAM) / 1000;
	c.beatsPerBar = int(p(BEATS_PER_BAR_PARAM));
	c.ratio = RATIOS[clamp(int(p(RATIO_PARAM)), 0, int(RATIOS.size()) - 1)];
	c.warmBpm = warmStart == WARM_TEMPO ? p(WARM_BPM_PARAM) : 0;
	c.seedFromClock = warmStart == WARM_CLOCK;
	c.clockPull = p(CLOCK_PULL_PARAM);
	c.tapPull = p(TAP_PULL_PARAM);
	c.autoDownbeat = autoDownbeat;
	c.nudgeRate = p(NUDGE_RATE_PARAM);
	return c;
}

double BeatDetector::bpmVoltage(double bpm) const {
	switch (bpmScale) {
		case BPM_LINEAR_300: return bpm / 30;
		case BPM_PER_100: return bpm / 100;
		default: return std::log2(std::max(bpm, 1.0) / 120);
	}
}

void BeatDetector::process(const ProcessArgs& args) {
	// The old inference is freed off the audio thread, so wait until the last one has been collected.
	if (!retired.load(std::memory_order_acquire))
		if (Swap* s = incoming.exchange(nullptr)) {
			s->next = engine->setInference(std::move(s->next));
			retired.store(s, std::memory_order_release);
		}
	if (configDivider.process() || pendingReset)
		engine->configure(settings());
	if (pendingReset.exchange(false))
		engine->reset();

	bool tap = tapButton.process(params[TAP_PARAM].getValue() > 0);
	tap |= tapIn.process(inputs[TAP_INPUT].getVoltage(), 0.1f, 1.f);
	if (tap)
		engine->tap();
	bool reset = resetButton.process(params[RESET_PARAM].getValue() > 0);
	reset |= resetIn.process(inputs[RESET_INPUT].getVoltage(), 0.1f, 1.f);
	if (reset) {
		engine->reset();
		clockCount = 0;
	}
	bool barReset = barResetButton.process(params[BAR_RESET_PARAM].getValue() > 0);
	barReset |= barResetIn.process(inputs[BAR_RESET_INPUT].getVoltage(), 0.1f, 1.f);
	if (barReset)
		engine->barReset();
	if (halfButton.process(params[HALF_PARAM].getValue() > 0))
		engine->scaleTempo(0.5);
	if (doubleButton.process(params[DOUBLE_PARAM].getValue() > 0))
		engine->scaleTempo(2);
	if (clockIn.process(inputs[CLOCK_INPUT].getVoltage(), 0.1f, 1.f) && clockCount++ % std::max(clockPpqn, 1) == 0)
		engine->clock();
	engine->setClockConnected(inputs[CLOCK_INPUT].isConnected());
	engine->setHold(inputs[HOLD_INPUT].getVoltage() >= 1.f);
	engine->setNudge((params[NUDGE_UP_PARAM].getValue() > 0) - (params[NUDGE_DOWN_PARAM].getValue() > 0));

	float l = inputs[AUDIO_L_INPUT].getVoltageSum(), r = inputs[AUDIO_R_INPUT].getVoltageSum();
	bool hasL = inputs[AUDIO_L_INPUT].isConnected(), hasR = inputs[AUDIO_R_INPUT].isConnected();
	float x = hasL && hasR ? (l + r) / 2 : hasL ? l : r;
	beat::Output o = engine->process(x / 5.f);
	if (engine->frameReady() && !frames.full())
		frames.push(engine->frame());

	double bpm = engine->bpm();
	float width = params[TRIGGER_WIDTH_PARAM].getValue() / 1000;
	float clockWidth = std::min(width, float(30 / std::max(bpm * engine->config().ratio, 1.0)));
	if (o.beat) {
		beatPulse.trigger(width);
		beatFlash.trigger(0.06f);
	}
	if (o.bar) {
		barPulse.trigger(width);
		barFlash.trigger(0.12f);
	}
	if (o.clock)
		clockPulse.trigger(clockWidth);

	outputs[BEAT_OUTPUT].setVoltage(beatPulse.process(args.sampleTime) ? 10.f : 0.f);
	outputs[BAR_OUTPUT].setVoltage(barPulse.process(args.sampleTime) ? 10.f : 0.f);
	outputs[CLOCK_OUTPUT].setVoltage(clockPulse.process(args.sampleTime) ? 10.f : 0.f);
	outputs[BEAT_PHASE_OUTPUT].setVoltage(10.f * o.beatPhase);
	outputs[BAR_PHASE_OUTPUT].setVoltage(10.f * o.barPhase);
	outputs[BPM_OUTPUT].setVoltage(bpmVoltage(bpm));
	outputs[LOCK_OUTPUT].setVoltage(engine->locked() ? 10.f : 0.f);
	outputs[CONFIDENCE_OUTPUT].setVoltage(10.f * engine->confidence());

	bool beatLit = beatFlash.process(args.sampleTime), barLit = barFlash.process(args.sampleTime);
	if (lightDivider.process()) {
		float dt = args.sampleTime * lightDivider.getDivision();
		lights[BEAT_LIGHT].setBrightnessSmooth(beatLit, dt);
		lights[BAR_LIGHT].setBrightnessSmooth(barLit, dt);
		lights[LOCK_LIGHT].setBrightness(engine->locked());
	}
}

void BeatDetector::onSampleRateChange(const SampleRateChangeEvent& e) {
	engine.reset(new beat::Engine(settings(), e.sampleRate));
	frames.clear();
	applyModel(e.sampleRate);
}

void BeatDetector::onReset(const ResetEvent& e) {
	Module::onReset(e);
	odf = int(beat::Odf::SuperFlux);
	lost = int(beat::Lost::Coast);
	bpmScale = BPM_VOCT;
	warmStart = WARM_OFF;
	clockPpqn = 1;
	autoDownbeat = false;
	zoom = 1;
	scheme = SCHEME_SPECTRUM;
	neural = true;
	modelJson.clear();
	modelName.clear();
	engine.reset(new beat::Engine(settings(), APP->engine->getSampleRate()));
	frames.clear();
	applyModel();
}

json_t* BeatDetector::dataToJson() {
	json_t* root = json_object();
	json_object_set_new(root, "odf", json_integer(odf));
	json_object_set_new(root, "lost", json_integer(lost));
	json_object_set_new(root, "bpmScale", json_integer(bpmScale));
	json_object_set_new(root, "warmStart", json_integer(warmStart));
	json_object_set_new(root, "clockPpqn", json_integer(clockPpqn));
	json_object_set_new(root, "autoDownbeat", json_boolean(autoDownbeat));
	json_object_set_new(root, "zoom", json_integer(zoom));
	json_object_set_new(root, "scheme", json_integer(scheme));
	json_object_set_new(root, "neural", json_boolean(neural));
	if (!modelJson.empty()) {  // a loaded model travels with the patch
		json_object_set_new(root, "modelName", json_string(modelName.c_str()));
		json_object_set_new(root, "modelJson", json_string(modelJson.c_str()));
	}
	return root;
}

void BeatDetector::dataFromJson(json_t* root) {
	auto integer = [&](const char* key, int& v, int lo, int hi) {
		if (json_t* j = json_object_get(root, key))
			v = clamp(int(json_integer_value(j)), lo, hi);
	};
	integer("odf", odf, 0, 3);
	integer("lost", lost, 0, 2);
	integer("bpmScale", bpmScale, 0, 2);
	integer("warmStart", warmStart, 0, 2);
	integer("clockPpqn", clockPpqn, 1, 96);
	integer("zoom", zoom, 0, int(ZOOMS.size()) - 1);
	integer("scheme", scheme, 0, 1);
	if (json_t* j = json_object_get(root, "autoDownbeat"))
		autoDownbeat = json_boolean_value(j);
	json_t* j = json_object_get(root, "neural");
	neural = j && json_boolean_value(j);  // patches from before 2.1 keep the ODF they were made with
	j = json_object_get(root, "modelJson");
	modelJson = j ? json_string_value(j) : "";
	j = json_object_get(root, "modelName");
	modelName = j ? json_string_value(j) : "";
	if (!applyModel()) {
		modelJson.clear();
		modelName.clear();
		applyModel();
	}
	pendingReset = true;  // picks up the loaded warm start
}

// ---------------------------------------------------------------------------------------------
// Panel

namespace layout {
// millimetres; the panel is 28HP
const float W = 142.24f;
const Vec display = {6.f, 10.5f}, displaySize = {130.24f, 43.f};
const float knobY = 67.f, buttonY = 84.f, inputY = 99.5f, outputY = 116.8f;
const float plateTop = 105.f, plateBottom = 122.6f;

inline float column(int i, int n) {
	return display.x + displaySize.x / n * (i + 0.5f);
}
} // namespace layout

/** Text labels, drawn into the panel's framebuffer so they follow the light/dark theme. */
struct PanelLabels : widget::Widget {
	struct Label {
		float x, y;
		std::string text;
		bool onPlate;
	};
	std::vector<Label> labels;

	void draw(const DrawArgs& args) override {
		std::shared_ptr<window::Font> font = APP->window->loadFont(asset::system("res/fonts/Nunito-Bold.ttf"));
		if (!font)
			return;
		bool dark = settings::preferDarkPanels;
		NVGcolor ink = dark ? nvgRGB(0xd8, 0xdb, 0xe0) : nvgRGB(0x2b, 0x2e, 0x33);
		NVGcolor plateInk = nvgRGB(0xf0, 0xf2, 0xf4);
		nvgFontFaceId(args.vg, font->handle);
		nvgTextAlign(args.vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
		nvgTextLetterSpacing(args.vg, 0.6f);

		nvgFontSize(args.vg, 10.5f);
		nvgFillColor(args.vg, ink);
		nvgText(args.vg, mm2px(layout::W / 2), mm2px(5.4f), "BEAT DETECTOR", nullptr);

		nvgFontSize(args.vg, 7.2f);
		for (const Label& l : labels) {
			nvgFillColor(args.vg, l.onPlate ? plateInk : ink);
			nvgText(args.vg, mm2px(l.x), mm2px(l.y), l.text.c_str(), nullptr);
		}
		nvgFontSize(args.vg, 6.f);
		nvgFillColor(args.vg, nvgTransRGBAf(ink, 0.6f));
		nvgText(args.vg, mm2px(layout::W / 2), mm2px(125.6f), "un1tz3r0", nullptr);
	}
};

/** A slider for a param that only lives in the context menu. */
struct ParamSlider : ui::Slider {
	explicit ParamSlider(Quantity* q) {
		quantity = q;
		box.size.x = 220.f;
	}
};

struct BeatDetectorWidget : ModuleWidget {
	explicit BeatDetectorWidget(BeatDetector* module) {
		using namespace layout;
		setModule(module);
		app::ThemedSvgPanel* panel = createPanel(asset::plugin(pluginInstance, "res/BeatDetector.svg"),
			asset::plugin(pluginInstance, "res/BeatDetector-dark.svg"));
		setPanel(panel);
		PanelLabels* labels = new PanelLabels;
		labels->box.size = panel->box.size;
		panel->fb->addChild(labels);
		auto label = [&](float x, float y, const char* text, bool onPlate = false) {
			labels->labels.push_back({x, y, text, onPlate});
		};

		addChild(createWidget<ThemedScrew>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ThemedScrew>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ThemedScrew>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ThemedScrew>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		Waterfall* waterfall = createWidget<Waterfall>(mm2px(display));
		waterfall->box.size = mm2px(displaySize);
		waterfall->module = module;
		addChild(waterfall);

		struct Knob {
			int param;
			const char* name;
			bool snap;
		};
		const Knob knobs[] = {{BeatDetector::FOLLOW_PARAM, "FOLLOW", false},
			{BeatDetector::PHASE_OFFSET_PARAM, "PHASE", false},
			{BeatDetector::TIME_OFFSET_PARAM, "TIME", false},
			{BeatDetector::RATIO_PARAM, "RATIO", true},
			{BeatDetector::BEATS_PER_BAR_PARAM, "BEATS/BAR", true}};
		for (int i = 0; i < 5; i++) {
			Vec pos = mm2px(Vec(column(i, 6), knobY));
			addParam(knobs[i].snap ? createParamCentered<RoundBlackSnapKnob>(pos, module, knobs[i].param)
								   : createParamCentered<RoundBlackKnob>(pos, module, knobs[i].param));
			label(column(i, 6), knobY - 7.6f, knobs[i].name);
		}
		addParam(createLightParamCentered<VCVLightBezel<WhiteLight>>(
			mm2px(Vec(column(5, 6), knobY)), module, BeatDetector::TAP_PARAM, BeatDetector::BEAT_LIGHT));
		label(column(5, 6), knobY - 7.6f, "TAP");

		const std::pair<int, const char*> buttons[] = {{BeatDetector::HALF_PARAM, "÷2"},
			{BeatDetector::DOUBLE_PARAM, "×2"},
			{BeatDetector::NUDGE_DOWN_PARAM, "NUDGE −"},
			{BeatDetector::NUDGE_UP_PARAM, "NUDGE +"},
			{BeatDetector::BAR_RESET_PARAM, "BAR RESET"},
			{BeatDetector::RESET_PARAM, "RESET"}};
		for (int i = 0; i < 6; i++) {
			addParam(createParamCentered<VCVButton>(mm2px(Vec(column(i, 6), buttonY)), module, buttons[i].first));
			label(column(i, 6), buttonY - 5.6f, buttons[i].second);
		}

		const char* inputNames[] = {"L / MONO", "R", "CLOCK", "TAP", "RESET", "BAR RST", "HOLD"};
		for (int i = 0; i < BeatDetector::INPUTS_LEN; i++) {
			addInput(createInputCentered<ThemedPJ301MPort>(mm2px(Vec(column(i, 7), inputY)), module, i));
			label(column(i, 7), inputY - 6.3f, inputNames[i]);
		}

		const char* outputNames[] = {"BEAT", "PHASE", "BAR", "BAR PH", "CLOCK", "BPM", "LOCK", "CONF"};
		for (int i = 0; i < BeatDetector::OUTPUTS_LEN; i++) {
			addOutput(createOutputCentered<ThemedPJ301MPort>(mm2px(Vec(column(i, 8), outputY)), module, i));
			label(column(i, 8), outputY - 6.6f, outputNames[i], true);
		}
		const std::pair<int, int> lights[] = {{BeatDetector::BAR_LIGHT, BeatDetector::BAR_OUTPUT},
			{BeatDetector::LOCK_LIGHT, BeatDetector::LOCK_OUTPUT}};
		addChild(createLightCentered<SmallLight<YellowLight>>(
			mm2px(Vec(column(lights[0].second, 8) + 5.3f, outputY - 4.6f)), module, lights[0].first));
		addChild(createLightCentered<SmallLight<GreenLight>>(
			mm2px(Vec(column(lights[1].second, 8) + 5.3f, outputY - 4.6f)), module, lights[1].first));
	}

	void step() override {
		if (BeatDetector* m = getModule<BeatDetector>())
			m->collect();
		ModuleWidget::step();
	}

	static void loadModel(BeatDetector* m) {
		osdialog_filters* filters = osdialog_filters_parse("RTNeural JSON:json");
		char* path = osdialog_file(OSDIALOG_OPEN, nullptr, nullptr, filters);
		osdialog_filters_free(filters);
		if (!path)
			return;
		std::string oldJson = m->modelJson, oldName = m->modelName;
		bool wasNeural = m->neural;
		m->modelJson = readFile(path);
		m->modelName = system::getFilename(path);
		m->neural = true;
		std::free(path);
		if (!m->applyModel()) {
			m->modelJson = oldJson;
			m->modelName = oldName;
			m->neural = wasNeural;
		}
	}

	void appendContextMenu(Menu* menu) override {
		BeatDetector* m = getModule<BeatDetector>();
		if (!m)
			return;
		// Submenus are built on hover, after this function returns, so capture m by value.
		auto slider = [m](Menu* into, int param) { into->addChild(new ParamSlider(m->paramQuantities[param])); };

		menu->addChild(new MenuSeparator);
		menu->addChild(createSubmenuItem("Tempo range", "", [=](Menu* sub) {
			sub->addChild(createMenuLabel("Changing these restarts acquisition"));
			slider(sub, BeatDetector::MIN_BPM_PARAM);
			slider(sub, BeatDetector::MAX_BPM_PARAM);
			sub->addChild(createMenuLabel("Preferred range, for half/double tempo"));
			slider(sub, BeatDetector::PREF_MIN_BPM_PARAM);
			slider(sub, BeatDetector::PREF_MAX_BPM_PARAM);
		}));
		menu->addChild(createSubmenuItem("Beat activation", m->neural ? "Neural" : "ODF", [=](Menu* sub) {
			sub->addChild(createCheckMenuItem("Neural network", "", [=]() { return m->neural; }, [=]() {
				m->neural = true;
				m->applyModel();
			}));
			sub->addChild(createCheckMenuItem("Onset detection function", "", [=]() { return !m->neural; }, [=]() {
				m->neural = false;
				m->applyModel();
			}));
			sub->addChild(new MenuSeparator);
			sub->addChild(createMenuLabel("Model: " + (m->modelJson.empty() ? std::string("built-in") : m->modelName)));
			sub->addChild(createMenuItem("Load RTNeural JSON…", "", [=]() { loadModel(m); }));
			sub->addChild(createMenuItem("Use the built-in model", "", [=]() {
				m->modelJson.clear();
				m->modelName.clear();
				m->neural = true;
				m->applyModel();
			}, m->modelJson.empty()));
			if (!m->modelError.empty())
				sub->addChild(createMenuLabel(m->modelError));
		}));
		menu->addChild(createIndexPtrSubmenuItem("Onset detection", {"SuperFlux", "Spectral flux", "Complex domain", "HFC"}, &m->odf));
		menu->addChild(createIndexPtrSubmenuItem("BPM CV scaling",
			{"V/oct, 0 V = 120 BPM", "Linear, 10 V = 300 BPM", "Linear, 1 V = 100 BPM"}, &m->bpmScale));
		menu->addChild(createSubmenuItem("Warm start", "", [=](Menu* sub) {
			sub->addChild(createIndexPtrSubmenuItem("Mode", {"Off (cold start)", "Tempo below", "Seed from Clock input"}, &m->warmStart));
			slider(sub, BeatDetector::WARM_BPM_PARAM);
			sub->addChild(createMenuLabel("Applies on Reset and at patch load"));
		}));
		menu->addChild(createSubmenuItem("On losing the beat", "", [=](Menu* sub) {
			sub->addChild(createIndexPtrSubmenuItem("Behaviour", {"Coast indefinitely", "Stop after N bars", "Keep following audio"}, &m->lost));
			slider(sub, BeatDetector::STOP_BARS_PARAM);
			slider(sub, BeatDetector::GLIDE_PARAM);
		}));
		menu->addChild(createSubmenuItem("Lock", "", [=](Menu* sub) { slider(sub, BeatDetector::LOCK_THRESHOLD_PARAM); }));
		menu->addChild(createSubmenuItem("Clock input", "", [=](Menu* sub) {
			slider(sub, BeatDetector::CLOCK_PULL_PARAM);
			sub->addChild(createSubmenuItem("Pulses per beat", string::f("%d", m->clockPpqn), [=](Menu* ppqn) {
				for (int n : {1, 2, 4, 8, 12, 24, 48, 96})
					ppqn->addChild(createCheckMenuItem(string::f("%d", n), "",
						[=]() { return m->clockPpqn == n; }, [=]() { m->clockPpqn = n; }));
			}));
		}));
		menu->addChild(createSubmenuItem("Tap and nudge", "", [=](Menu* sub) {
			slider(sub, BeatDetector::TAP_PULL_PARAM);
			slider(sub, BeatDetector::NUDGE_RATE_PARAM);
		}));
		menu->addChild(createSubmenuItem("Outputs", "", [=](Menu* sub) { slider(sub, BeatDetector::TRIGGER_WIDTH_PARAM); }));
		menu->addChild(createBoolPtrMenuItem("Automatic downbeat", m->neural ? "network" : "low band", &m->autoDownbeat));
		menu->addChild(createSubmenuItem("Display", "", [=](Menu* sub) {
			sub->addChild(createIndexPtrSubmenuItem("Zoom", {"3 s", "6 s", "12 s", "24 s"}, &m->zoom));
			sub->addChild(createIndexPtrSubmenuItem("Colours", {"Spectrum (low warm, mid green, high blue)", "Mono"}, &m->scheme));
		}));
		menu->addChild(createSubmenuItem("Advanced", "", [=](Menu* sub) {
			for (int id = BeatDetector::ADVANCED_FIRST; id < BeatDetector::PARAMS_LEN; id++)
				slider(sub, id);
			sub->addChild(createMenuItem("Restore advanced defaults", "", [=]() {
				for (int id = BeatDetector::ADVANCED_FIRST; id < BeatDetector::PARAMS_LEN; id++)
					m->paramQuantities[id]->reset();
			}));
		}));
	}
};

Model* modelBeatDetector = createModel<BeatDetector, BeatDetectorWidget>("BeatDetector");

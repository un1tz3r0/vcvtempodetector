#pragma once
#include <memory>

#include "dsp/Engine.hpp"
#include "plugin.hpp"

struct BeatDetector : Module {
	enum ParamId {
		FOLLOW_PARAM,
		PHASE_OFFSET_PARAM,
		TIME_OFFSET_PARAM,
		RATIO_PARAM,
		BEATS_PER_BAR_PARAM,
		TAP_PARAM,
		HALF_PARAM,
		DOUBLE_PARAM,
		NUDGE_DOWN_PARAM,
		NUDGE_UP_PARAM,
		BAR_RESET_PARAM,
		RESET_PARAM,
		// context menu
		MIN_BPM_PARAM,
		MAX_BPM_PARAM,
		PREF_MIN_BPM_PARAM,
		PREF_MAX_BPM_PARAM,
		WARM_BPM_PARAM,
		LOCK_THRESHOLD_PARAM,
		GLIDE_PARAM,
		STOP_BARS_PARAM,
		CLOCK_PULL_PARAM,
		TAP_PULL_PARAM,
		NUDGE_RATE_PARAM,
		TRIGGER_WIDTH_PARAM,
		// context menu, advanced
		LATENCY_PARAM,
		LOW_WEIGHT_PARAM,
		GAMMA_PARAM,
		TRANSITION_LAMBDA_PARAM,
		OBSERVATION_LAMBDA_PARAM,
		TEMPO_PRIOR_PARAM,
		CONF_TAU_PARAM,
		CONF_SCALE_PARAM,
		SILENCE_PARAM,
		PARAMS_LEN
	};
	enum InputId { AUDIO_L_INPUT, AUDIO_R_INPUT, CLOCK_INPUT, TAP_INPUT, RESET_INPUT, BAR_RESET_INPUT, HOLD_INPUT, INPUTS_LEN };
	enum OutputId {
		BEAT_OUTPUT,
		BEAT_PHASE_OUTPUT,
		BAR_OUTPUT,
		BAR_PHASE_OUTPUT,
		CLOCK_OUTPUT,
		BPM_OUTPUT,
		LOCK_OUTPUT,
		CONFIDENCE_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId { BEAT_LIGHT, BAR_LIGHT, LOCK_LIGHT, LIGHTS_LEN };

	enum BpmScale { BPM_VOCT, BPM_LINEAR_300, BPM_PER_100 };
	enum WarmStart { WARM_OFF, WARM_TEMPO, WARM_CLOCK };
	enum Scheme { SCHEME_SPECTRUM, SCHEME_MONO };

	static constexpr int ADVANCED_FIRST = LATENCY_PARAM;
	static const std::vector<float> RATIOS;
	static const std::vector<std::string> RATIO_LABELS;
	static const std::vector<float> ZOOMS;

	// context menu settings that aren't params
	int odf = int(beat::Odf::SuperFlux);
	int lost = int(beat::Lost::Coast);
	int bpmScale = BPM_VOCT;
	int warmStart = WARM_OFF;
	int clockPpqn = 1;
	bool autoDownbeat = false;
	int zoom = 1;
	int scheme = SCHEME_SPECTRUM;

	// beat activation: a neural network (the built-in one, or a loaded RTNeural JSON kept in the
	// patch), or the onset detection function
	bool neural = true;
	std::string modelJson, modelName;  // empty: the built-in model
	std::string modelError;            // why the last model didn't load, for the menu

	std::unique_ptr<beat::Engine> engine;
	dsp::RingBuffer<beat::Frame, 512> frames;  // audio thread -> display
	std::atomic<bool> pendingReset{false};

	BeatDetector();
	~BeatDetector();
	beat::Config settings();
	/** Build the network for the current choice and hand it to the audio thread; false (and
	modelError set) if the model can't be used, in which case the activation is left as it was. */
	bool applyModel(double sampleRate = 0);
	/** Delete what the audio thread has swapped out. Never on the audio thread. */
	void collect();
	double bpmVoltage(double bpm) const;

	void process(const ProcessArgs& args) override;
	void onSampleRateChange(const SampleRateChangeEvent& e) override;
	void onReset(const ResetEvent& e) override;
	json_t* dataToJson() override;
	void dataFromJson(json_t* root) override;

private:
	struct Swap {
		std::unique_ptr<beat::Inference> next;  // nullptr: back to the ODF
	};
	std::atomic<Swap*> incoming{nullptr}, retired{nullptr};

	dsp::ClockDivider configDivider, lightDivider;
	dsp::SchmittTrigger tapIn, clockIn, resetIn, barResetIn;
	dsp::BooleanTrigger tapButton, halfButton, doubleButton, barResetButton, resetButton;
	dsp::PulseGenerator beatPulse, barPulse, clockPulse, beatFlash, barFlash;
	int clockCount = 0;
};

#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include <vector>
#include <string>
#include "PID.h"

#define NUM_INPUT 8
#define NUM_ACTIONS 3
#define NUM_HIDDEN 256

struct EventData { 

	bool done;
	double Obj; // The X or Y center of object on in frame
	double size; // The bounding size of object along its X or Y axis
	double point; // The X or Y Origin coordinate of object
	double Frame; // The X or Y center of frame
	double error;
	double timestamp;
	void reset() {
		done = true;
		error = 0.0;
		Obj = 0.0;
		size = 0.0;
		Frame = 0.0;
		timestamp = 0.0;
	}
	EventData() : done(true), error(0.0), Frame(0.0), Obj(0.0), timestamp(0.0) {}
} typedef ED;

struct StateData {

	double stateArray[NUM_INPUT];

	void setStateArray(double state[NUM_INPUT]) {
		for (int i = 0; i < NUM_INPUT; i++) {
			stateArray[i] = state[i];
		}
	}

	void getStateArray(double state[NUM_INPUT]) {

		for (int i = 0; i < NUM_INPUT; i++) {
			state[i] = stateArray[i];
		}
	}

} typedef SD;

struct StepResults {
	SD servos[2];
} typedef SR;

struct TrainData {
	SD currentState;
	SD nextState;
	double reward;
	double actions[NUM_ACTIONS];
	bool done;
} typedef TD;

typedef std::vector<TD> TrainBuffer;

struct Config {


	// Network Options
	int numActions;
	int numHidden;
	int numInput;

	// Train Options
	int maxBufferSize;
	int minBufferSize;
	int maxTrainingSessions;
	int batchSize;
	bool offPolicyTrain;
	bool initialRandomActions;
	int numInitialRandomActions;
	bool trainMode;
	bool defaultMode;
	bool frameSkip;
	int numFrameSkip;

	// Tracking Options
	float recheckChance;
	int trackerType;
	bool useTracking;
	bool useAutoTuning;
	bool draw;
	bool showVideo;
	bool cascadeDetector;
	int lossCountMax;
	std::string target;

	// Servo options
	double angleHigh;
	double angleLow;
	double resetAngleX;
	double resetAngleY;
	double pidOutputHigh;
	double pidOutputLow;
	double defaultGains[3];
	int updateRate;
	double trainRate;
	bool invertX;
	bool invertY;
	bool disableX;
	bool disableY;
	bool useArduino;
	unsigned char arduinoCommands[3]; 

	// bounds
	double actionHigh;
	double actionLow;

	Config() :

		numActions(NUM_ACTIONS),
		numHidden(NUM_HIDDEN),
		numInput(NUM_INPUT),

		maxBufferSize(1e5),
		minBufferSize(1e3),
		maxTrainingSessions(1),
		batchSize(128),
		initialRandomActions(true),
		numInitialRandomActions(5e3),
		trainMode(true), // When autotuning is on, use to execute means from network as PID gains and save to replay buffer..
		useAutoTuning(true), // Use SAC network to query for PID gains
		frameSkip(false),
		numFrameSkip(4),

		recheckChance(0.2),
		lossCountMax(2),
		updateRate(10),
		trainRate(0.08),
		invertX(false),
		invertY(true),
		disableX(false),
		disableY(true),

		useArduino(false),
		arduinoCommands({ 0x3, 0x2, 0x8 }),

		trackerType(1),
		useTracking(true),
		draw(false),
		showVideo(false),
		cascadeDetector(true),
		target("face"),
		actionHigh(0.1),
		actionLow(0.0001),
		pidOutputHigh(70.0),
		pidOutputLow(-70.0),
		defaultGains({0.05, 0.04, 0.001}),
		// defaultGains({ 0.0, 0.0, 0.0 }),
		angleHigh(70.0),
		angleLow(-70.0),
		resetAngleX(0.0),
		resetAngleY(-15.0),
		offPolicyTrain(true)
		{}
} typedef cfg;

struct Parameter {
	PID* pan;
	PID* tilt;

	int dims[2];

	int pid;
	ED* eventData;
	int rate; // Updates per second
	int fd;

	bool isTraining;
	bool freshData;
	cfg* config;
	
} typedef param;

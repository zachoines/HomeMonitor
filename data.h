#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include <vector>
#include <string>
#include "PID.h"

#define NUM_SERVOS 2
#define NUM_INPUT 5
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
		done = false;
		error = 0.0;
		Obj = 0.0;
		size = 0.0;
		Frame = 0.0;
		timestamp = 0.0;
	}
	EventData() : done(false), error(0.0), Frame(0.0), Obj(0.0), timestamp(0.0) {}
} typedef ED;

// Current state of servo/pid
struct PIDAndServoStateData {
	
	struct PIDState pidStateData;
	double currentAngle;
	double lastAngle;
	double obj;

	void getStateArray(double state[NUM_INPUT]) {
		pidStateData.getStateArray(state); // first 4 elems will be filled
		state[4] = lastAngle - currentAngle;
	}

} typedef SD;

// State of PIDs/Servos after reset
struct ResetData {
	SD servos[NUM_SERVOS];
} typedef RD;

// Result Data for taking a step with for a given servo/PID
struct TrainData {
	SD currentState;
	SD nextState;
	double reward;
	double actions[NUM_ACTIONS];
	bool done;
} typedef TD;

// Results of taking actions for each PID/servos in env
struct StepResults {
	TD servos[NUM_SERVOS];
} typedef SR;

typedef std::vector<TD> TrainBuffer;

struct Config {


	// Network Options
	int numActions;
	int numHidden;
	int numInput;

	// Train Options
	int maxBufferSize;
	int minBufferSize;
	long maxTrainingSteps;
	int maxTrainingSessions;
	double numUpdates;
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

		numActions(NUM_ACTIONS),             // Number of output classifications for policy network
		numHidden(NUM_HIDDEN),               // Number of nodes in the hidden layers of each network
		numInput(NUM_INPUT),                 // Number of elements in policy/value network's input vectors

		maxBufferSize(1000000),              // Max size of buffer. When full, oldest elements are kicked out.
		minBufferSize(2000),                 // Min replay buffer size before training size.
		maxTrainingSteps(1000000),			 // Max training steps agent takes.
		maxTrainingSessions(1),              // TODO: Remove this option
		numUpdates(5),                       // Num updates per training session.

		batchSize(256),                      // Network batch size.
		initialRandomActions(true),          // Enable random actions.
		numInitialRandomActions(20000),      // Number of random actions taken.
		trainMode(true),                     // When autotuning is on, 'false' means network test mode.
		useAutoTuning(true),                 // Use SAC network to query for PID gains.
		frameSkip(false),                    // TODO: Remove this option
		numFrameSkip(4),                     // TODO: Remove this option

		recheckChance(0.2),                  // Chance to revalidate tracking quality
		lossCountMax(5),                     // Max number of rechecks before episode is considered over
		updateRate(5),                       // Servo updates, commands per second
		trainRate(0.5),					     // Network updates, sessions per second
		invertX(false),                      // Flip output angles for pan
		invertY(true),						 // Flip output angles for tilt
		disableX(false),                     // Disable the pan servo
		disableY(true),                      // Disable the tilt servo

		useArduino(false),					 // TODO: Remove this option
		arduinoCommands({ 0x3,				 // I2c commands end to arduino
			0x2, 
			0x8 
		}),

		trackerType(1),						 // { CSRT, MOSSE, GOTURN } 
		useTracking(true),					 // Use openCV tracker instead of face detection
		draw(false),						 // Draw target bounding box and center on frame
		showVideo(false),					 // Show camera feed
		cascadeDetector(true),				 // Use faster cascade face detector 
		target("face"),						 // TODO: remove this option
		actionHigh(0.07),                     // Max output to of policy network's logits
		actionLow(0.0),                      // Min output to of policy network's logits        
		pidOutputHigh(70.0),                 // Max output allowed for PID's
		pidOutputLow(-70.0),				 // Min output allowed for PID's
		// defaultGains({0.11, 0.09, 0.002}),
		defaultGains({ 0.05, 0.04, 0.001 }), // Gains fed to pids when initialized
		// defaultGains({ 1.0, 1.0, 1.0 }), 
		angleHigh(70.0),                     // Max allowable output angle to servos
		angleLow(-70.0),                     // Min allowable output angle to servos
		resetAngleX(-1.0),                   // Angle when reset
		resetAngleY(-5.0),                   // Angle when reset
		offPolicyTrain(true)                 // Use uniform random sampling from replay buffer without removal
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

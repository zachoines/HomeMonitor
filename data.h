#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include <vector>
#include "PID.h"

//#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/containers/vector.hpp>
//#include <boost/interprocess/allocators/allocator.hpp>

struct EventData {

	bool done;
	double Obj;
	double Frame;
	double error;
	double timestamp;

	EventData() : done(false) {}
} typedef ED;

struct StateData {

	double stateArray[3];
	
	double objCenter;
	double frameCenter;
	double error;

	double objCenterOld;
	double frameCenterOld;
	double errorOld;
	
	double p;
	double i;
	double d;

	void setStateArray(double state[3]) {
		for (int i = 0; i < 3; i++) {
			stateArray[i] = state[i];
		}
	}

	void getStateArray(double state[3]) {

		for (int i = 0; i < 3; i++) {
			state[i] = stateArray[i];
		}
	}

} typedef SD;

struct TrainData {
	SD currentState;
	SD nextState;
	double reward;
	double actions[1];
	bool done;
} typedef TD;

// typedef boost::interprocess::allocator<TD, boost::interprocess::managed_shared_memory::segment_manager> ShmemAllocator;
// typedef boost::interprocess::vector<TD, ShmemAllocator> SharedBuffer;
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

	// Tracking Options
	float recheckChance;
	int trackerType;
	bool useTracking;
	bool draw;
	bool showVideo;
	bool cascadeDetector;
	int lossCountMax;
	std::string target;

	// Servo options
	double angleHigh;
	double angleLow;
	int updateRate;
	bool invertX;
	bool invertY;
	bool useArduino;
	unsigned char arduinoCommands[3];

	// bounds
	double actionHigh;
	double actionLow;

	Config() :

		numActions(3),
		numHidden(128),
		numInput(3),

		maxBufferSize(100000),
		minBufferSize(128),
		maxTrainingSessions(32),
		batchSize(32),

		recheckChance(0.05),
		lossCountMax(10),
		updateRate(10),
		invertX(false),
		invertY(true),

		useArduino(false),
		arduinoCommands({ 0x3, 0x2, 0x8 }),

		trackerType(1),
		useTracking(true),
		draw(false),
		showVideo(false),
		cascadeDetector(true),
		target("face"),

		actionHigh(10.0),
		actionLow(0.0),
		angleHigh(165.0),
		angleLow(15.0),
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

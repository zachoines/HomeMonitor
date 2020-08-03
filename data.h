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

//struct StateData {
//	double objCenter;
//	double frameCenter;
//	double error;
//	double currentAngle;
//	double objCenterOld;
//	double frameCenterOld;
//	double errorOld;
//	double lastAngle;
//
//	double getStateArray(double state[8]) {
//		state[0] = objCenter;
//		state[1] = frameCenter;
//		state[2] = error;
//		state[3] = currentAngle;
//		state[4] = objCenterOld;
//		state[5] = frameCenterOld;
//		state[6] = errorOld;
//		state[7] = lastAngle;
//	}
//
//} typedef SD;


struct StateData {
	
	double objCenter;
	double frameCenter;
	double error;

	double objCenterOld;
	double frameCenterOld;
	double errorOld;
	
	double p;
	double i;
	double d;

	double getStateArray(double state[7]) {
		state[0] = objCenter;
		state[1] = frameCenter;
		// state[2] = error;

		state[2] = objCenterOld;
		state[3] = frameCenterOld;
		// state[5] = errorOld;

		state[4] = p;
		state[5] = i;
		state[6] = d;
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

	// Tracking Options
	float recheckChance;
	int trackerType;
	bool useTracking;
	bool draw;
	bool showVideo;
	bool cascadeDetector;
	int lossCountMax;
	std::string target;

	// bounds
	double actionHigh;
	double actionLow;

	Config() :
		numActions(1),
		numHidden(7),
		numInput(7),
		maxBufferSize(1024),
		minBufferSize(128),
		maxTrainingSessions(32),
		batchSize(32),
		recheckChance(0.05),
		trackerType(1),
		useTracking(true),
		draw(false),
		showVideo(false),
		cascadeDetector(true),
		lossCountMax(20),
		target("face"),
		actionHigh(165.0),
		actionLow(15.0)
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

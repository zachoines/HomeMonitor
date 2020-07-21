#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include <vector>
#include "PID.h"
// #include "Model.h"
//#include <boost/interprocess/managed_shared_memory.hpp>
//#include <boost/interprocess/containers/vector.hpp>
//#include <boost/interprocess/allocators/allocator.hpp>


/* eventData allows for async communication between parent processand child process's threads
   These garentee, in a non-blocking way, the child process will never read a value twice.
   But the child's threads may be reading old values at any given time. */
struct EventData {
	bool isOld(double t) {
		return (timestamp - t == 0.0);
	};

	bool dirty;
	bool done;
	double Obj;
	double Frame;
	double error;
	double timestamp;

	EventData() : dirty(false), done(false) {}
} typedef ED;

struct StateData {
	double objCenter;
	double frameCenter;
	double error;
	double currentAngle;
	double objCenterOld;
	double frameCenterOld;
	double errorOld;
	double lastAngle;

	double getStateArray(double state[8]) {
		state[0] = objCenter;
		state[1] = frameCenter;
		state[2] = error;
		state[3] = currentAngle;
		state[4] = objCenterOld;
		state[5] = frameCenterOld;
		state[6] = errorOld;
		state[7] = lastAngle;
	}

} typedef SD;

struct TrainData {
	SD currentState;
	SD nextState;
	double reward;
	double actions[3];
	bool done;
} typedef TD;

// typedef boost::interprocess::allocator<TD, boost::interprocess::managed_shared_memory::segment_manager> ShmemAllocator;
// typedef boost::interprocess::vector<TD, ShmemAllocator> SharedBuffer;
typedef std::vector<TD> Buffer;

struct Config {

	
	int numActions;
	int numHidden;
	int numInput;
	int maxBufferSize;
	int maxTrainingSessions;
	int batchSize;

	Config() : 
		numActions(1), 
		numHidden(256), 
		numInput(8), 
		maxBufferSize(128), 
		maxTrainingSessions(4), 
		batchSize(32) {}
} typedef cfg;

struct Parameter {
	PID* pan;
	PID* tilt;
	int height;
	int width;

	int pid;
	ED* ShmPTR;
	int rate; // Updates per second
	int fd;
	pthread_mutex_t mutex;

	bool isTraining;
	cfg* config;
	
} typedef param;

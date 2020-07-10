#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include "PID.h"
// #include "Model.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

// These values allow for async communication between parent processand child process's threads
// These garentee, in a non-blocking way, the child process will never read a value twice.
// But the child's threads may be reading old values at any given time.
struct eventData {

	eventData() : dirty(false), old(false), done(false) {}

	bool isOld(double t) {
		return (timestamp - t == 0.0);
	};

	bool dirty;
	bool old;
	bool done;
	double Obj;
	double Frame;
	double Angle;
	double error;
	double timestamp;

} typedef ED;

struct stateData {
	double Obj;
	double Frame;
	double Angle;
	double error;
	double stateArray[4] = {
		this->Obj,
		this->Frame,
		this->Angle,
		this->error
	};
} typedef SD;

struct TrainData {
	SD currentState;
	SD nextState;
	double reward;
	double actions[3];
	bool done;
} typedef TD;

typedef boost::interprocess::allocator<TD, boost::interprocess::managed_shared_memory::segment_manager> ShmemAllocator;
typedef boost::interprocess::vector<TD, ShmemAllocator> Buffer;

struct parameter {
	PID* pan;
	PID* tilt;

	int pid;
	ED* ShmPTR;
	int rate; // Updates per second
	int fd;
	pthread_mutex_t mutex;

	bool isTraining;
	int maxBufferSize;
} typedef param;

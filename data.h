#pragma once
/*
	This file should contain data formats, structs, unions, and typedefs.
	Keeps main.cpp cleaner.
*/
#include "PID.h"
#include "Model.h"
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>


struct TrainData {
	double outputGains[3];
	double error;
} typedef TD;

typedef boost::interprocess::allocator<TD, boost::interprocess::managed_shared_memory::segment_manager> ShmemAllocator;
typedef boost::interprocess::vector<TD, ShmemAllocator> Buffer;

struct parameter {
	PID* pan;
	PID* tilt;
	PIDAutoTuner* model;

	int pid;
	int* ShmPTR;
	int rate; // Updates per second
	int fd;
	pthread_mutex_t mutex;

	bool isTraining;
	int maxBufferSize;
} typedef param;


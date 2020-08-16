#pragma once
#include <vector>
#include <random>
#include <iterator>
#include "util.h"
#include "data.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

class ReplayBuffer
{
private:
	pthread_mutex_t _trainBufferLock;
	TrainBuffer* _trainingBuffer;
	int _maxBufferSize;
	int _bufferIndex = -1;

	// random number generation
	std::random_device _myRandomDevice;
	std::default_random_engine _myRandomEngine;

public:
	ReplayBuffer(int maxBufferSize = 1024);
	TrainBuffer sample(int batchSize = 32);
	TrainBuffer getCopy();
	void add(TD data);
	int size();
	void clear();
};


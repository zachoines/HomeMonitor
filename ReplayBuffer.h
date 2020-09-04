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

	// Create a random device and use it to generate a random seed
	std::mt19937 eng{ std::random_device{}() };

public:
	ReplayBuffer(int maxBufferSize = 1024);
	TrainBuffer sample(int batchSize = 32, bool remove = true);
	TrainBuffer getCopy();
	void add(TD data);
	int size();
	void clear();
	int _draw(int min, int max);
};


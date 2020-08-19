#pragma once
#include <vector>
#include <random>
#include <iterator>
#include "util.h"
#include "data.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>


class ReplayBuffer
{
private:
	pthread_mutex_t _trainBufferLock;
	SharedBuffer* _trainingBuffer;
	int _maxBufferSize;
	int _bufferIndex = -1;

	// random number generation
	std::random_device _myRandomDevice;
	std::default_random_engine _myRandomEngine;

	boost::interprocess::named_mutex* _mutex;

public:
	ReplayBuffer(int maxBufferSize, SharedBuffer* buffer);
	~ReplayBuffer();
	TrainBuffer sample(int batchSize = 32);
	void add(TD data);
	void removeOld(int size);
	int size();
	void clear();
};


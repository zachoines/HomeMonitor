#include "ReplayBuffer.h"
#include <vector>
#include <random>
#include <iterator>
#include "data.h"
#include "util.h"

// Boost imports
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

ReplayBuffer::ReplayBuffer(int maxBufferSize, SharedBuffer * buffer)
{

	_trainingBuffer = buffer;

	// Setup interprocess mutex
	_mutex = new boost::interprocess::named_mutex(boost::interprocess::open_or_create, "ReplayBufferMutex");

	_bufferIndex = _trainingBuffer->size() % maxBufferSize;

	_maxBufferSize = maxBufferSize;

	// Create a random device and use it to generate a random seed
	unsigned seed = _myRandomDevice();

	// Initialize a default_random_engine with the seed
	_myRandomEngine.seed(seed);
}

ReplayBuffer::~ReplayBuffer() {
	boost::interprocess::named_mutex::remove("ReplayBufferMutex");
	delete _mutex;
}

TrainBuffer ReplayBuffer::sample(int batchSize)
{
	TrainBuffer batch;

	// Initialize a uniform_int_distribution to produce values
	std::uniform_int_distribution<int> myUnifIntDist(0, _trainingBuffer->size() - 1);

	// Randomly generated values
	boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
	for (int i = 0; i < batchSize; i++) {
		int number = myUnifIntDist(_myRandomEngine);
		batch.push_back(_trainingBuffer->at(number));
	}


	return batch;
}

void ReplayBuffer::removeOld(int size) {
	boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);

	if (_trainingBuffer->size() < size) {
		_trainingBuffer->clear();
	}
	auto first = _trainingBuffer->begin() + 0;
	auto last = _trainingBuffer->begin() + size + 1;
	_trainingBuffer->erase(first, last);
	_trainingBuffer->shrink_to_fit();
}

void ReplayBuffer::add(TD data)
{
	boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
	_bufferIndex = (_bufferIndex + 1) % _maxBufferSize;
	if (_trainingBuffer->size() == _maxBufferSize) {
		_trainingBuffer->at(_bufferIndex) = data;
	}
	else {
		_trainingBuffer->push_back(data);
	}
}

int ReplayBuffer::size() 
{
	boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
	return _trainingBuffer->size();
}

void ReplayBuffer::clear() 
{
	boost::interprocess::scoped_lock<boost::interprocess::named_mutex> lock(*_mutex);
	_trainingBuffer->clear();

}
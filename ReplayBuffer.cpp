#include "ReplayBuffer.h"
#include <vector>
#include <random>
#include <iterator>

#include <cstdlib>
#include <ctime>
#include <iostream>


ReplayBuffer::ReplayBuffer(int maxBufferSize)
{
	_trainingBuffer = new TrainBuffer();
	_trainBufferLock = PTHREAD_MUTEX_INITIALIZER;
	_maxBufferSize = maxBufferSize;

	
}

int ReplayBuffer::_draw(int min, int max)
{
	return std::uniform_int_distribution<int>{min, max}(eng);
}

TrainBuffer ReplayBuffer::sample(int batchSize, bool remove)
{
	TrainBuffer batch;

	if (batchSize > _trainingBuffer->size()) {
		throw "Batchsize cannot be larger than buffer size";
	}

	if (pthread_mutex_lock(&_trainBufferLock) == 0) {

		for (int i = 0; i < batchSize; i++) {
			int number = _draw(0, _trainingBuffer->size() - 1);
			
			batch.push_back(_trainingBuffer->at(number));

			if (remove) {
				_trainingBuffer->erase(_trainingBuffer->begin() + number);
				_bufferIndex = (_bufferIndex - 1) % _maxBufferSize;
			}
		}
	}

	pthread_mutex_unlock(&_trainBufferLock);
	return batch;
}

TrainBuffer ReplayBuffer::getCopy() {

	TrainBuffer buff;
	if (pthread_mutex_lock(&_trainBufferLock) == 0) {
		buff = *_trainingBuffer;
	}

	pthread_mutex_unlock(&_trainBufferLock);

	return buff;
}

void ReplayBuffer::add(TD data)
{
	_bufferIndex = (_bufferIndex + 1) % _maxBufferSize;
	if (pthread_mutex_lock(&_trainBufferLock) == 0) {
		if (_trainingBuffer->size() == _maxBufferSize) {
			_trainingBuffer->at(_bufferIndex) = data;
		}
		else {
			_trainingBuffer->push_back(data);
		}
	}

	pthread_mutex_unlock(&_trainBufferLock);
}

int ReplayBuffer::size() {
	return _trainingBuffer->size();
}

void ReplayBuffer::clear() {
	
	if (pthread_mutex_lock(&_trainBufferLock) == 0) {
		_trainingBuffer->clear();
	}
	_bufferIndex = -1;
	
	pthread_mutex_unlock(&_trainBufferLock);
}
#include "ReplayBuffer.h"
#include <vector>
#include <random>
#include <iterator>


ReplayBuffer::ReplayBuffer(int maxBufferSize)
{
	_trainingBuffer = new TrainBuffer();
	_trainBufferLock = PTHREAD_MUTEX_INITIALIZER;
	_maxBufferSize = maxBufferSize;

	// Create a random device and use it to generate a random seed
	unsigned seed = _myRandomDevice();

	// Initialize a default_random_engine with the seed
	_myRandomEngine.seed(seed);
}

TrainBuffer ReplayBuffer::sample(int batchSize)
{
	TrainBuffer batch;

	if (pthread_mutex_lock(&_trainBufferLock) == 0) {

		// Initialize a uniform_int_distribution to produce values
		std::uniform_int_distribution<int> myUnifIntDist(0, _trainingBuffer->size() - 1);

		// Create and print 5 randomly generated values
		for (int i = 0; i < batchSize; i++) {
			int number = myUnifIntDist(_myRandomEngine);
			batch.push_back(_trainingBuffer->at(number));
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
	
	pthread_mutex_unlock(&_trainBufferLock);
}

/*while (!parameters->freshData) {
			pthread_cond_wait(&trainBufferCond, &trainBufferLock);
		}*/

		//if (offPolicy) {
		//	// Shrink working copy if too large
		//	if (trainingBufferCopy.size() == maxBufferSize) {
		//		erase(trainingBufferCopy, 0, minBufferSize - 1);
		//	}

		//	// Add elements to our working copy
		//	trainingBufferCopy = append(trainingBufferCopy, *trainingBuffer);
		//	trainingBuffer->clear();
		//	parameters->freshData = false;
		//	pthread_mutex_unlock(&trainBufferLock);

		//	// retrieve random batch
		//	// std::random_shuffle(trainingBufferCopy.begin(), trainingBufferCopy.end());
		//	for (int i = 0; i < batchSize; i++) {

		//	}

		//	// Perform training session
		//	for (int i = 0, m = 0; i < sessions, m < maxBufferSize; i++, m += batchSize - 1) {

		//		// Generate training sample
		//		int n = m + batchSize - 1;
		//		if (n < trainingBufferCopy.size()) {
		//			TrainBuffer subBuf = slice(trainingBufferCopy, m, n);
		//			pidAutoTuner->update(batchSize, &subBuf);
		//		}
		//		else {
		//			parameters->isTraining = false;
		//			break;
		//		}
		//	}
		//} 
		//else {

		//	trainingBufferCopy.clear();
		//	trainingBufferCopy = append(trainingBufferCopy, *trainingBuffer);
		//	trainingBuffer->clear();
		//	parameters->freshData = false;
		//	pthread_mutex_unlock(&trainBufferLock);

		//	pidAutoTuner->update(batchSize, &trainingBufferCopy);

		//	parameters->isTraining = false;
		//}
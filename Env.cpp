#include "Env.h"
#include "data.h"
#include "util.h"
#include <cmath>

Env::Env(param* parameters, pthread_mutex_t* stateDataLock, pthread_cond_t* stateDataCond)
{
	_cond = stateDataCond;
	_lock = stateDataLock;
	_params = parameters;
	_config = parameters->config;

	_resetAngles[0] = parameters->config->resetAngleY;
	_resetAngles[1] = parameters->config->resetAngleX;

	_disableServo[0] = parameters->config->disableY;
	_disableServo[1] = parameters->config->disableX;

	_pids[0] = parameters->tilt;
	_pids[1] = parameters->pan;

	_currentAngles[0] = 0.0;
	_currentAngles[1] = 0.0;

	_lastAngles[0] = 0.0;
	_lastAngles[1] = 0.0;

	_errorCounter = -1;

	_invert[0] = _params->config->invertY;
	_invert[1] = _params->config->invertX;

}

void Env::_updateState()
{

	for (int i = 0; i < 2; i++) {
		if (_disableServo[i]) {
			continue;
		}
		if (_currentData[i].done) {
			resetEnv();
		}

		// Normalize and fill out the current state
		double scale = static_cast<double>(_params->dims[i]) / 2.0;
		double state[_params->config->numInput];

		_lastTimeStamp[i] = _currentData[i].timestamp;
		
		double gains[3] = {
			0.0
		};

		_pids[i]->getPID(gains);

		// Last angle history
		state[0] = _currentAngles[i] / _params->config->angleHigh;
		state[1] = _lastAngles[i] / _params->config->angleHigh;

		// Last error history
		state[2] = (_currentData[i].Obj - (static_cast<double>(_params->dims[i]) / 2.0)) / scale;
		state[3] = (_lastData[i].Obj - (static_cast<double>(_params->dims[i]) / 2.0)) / scale;

		// Integral and dirivative from PID
		state[4] = gains[1] / scale;
		state[5] = gains[2] / scale;

		// Last target locations
		state[6] = _currentData[i].Obj / scale;
		state[7] = _lastData[i].Obj / scale;

		_lastState[i] = _currentState[i];
		_currentState[i].setStateArray(state);

		if (!_lastData[i].done) {
			_hasData = true;
		}
	}
}

void Env::_sleep()
{
	int milis = 1000 / _params->rate;
	Utility::msleep(milis);
}

bool Env::init()
{
	if (pthread_mutex_lock(_lock) == 0) {
		for (int i = 0; i < 2; i++) {
			if (_disableServo[i]) {
				continue;
			}
			_lastReward[i] = 0.0;
			_lastData[i] = _currentData[i];
			_currentData[i] = _params->eventData[i];
		}
		pthread_mutex_unlock(_lock);
	}
	else {
		return false;
	}

	_updateState();

	return true;
}

void Env::ping()
{
	_resetData();
	_syncEnv();
	_updateState();
}

void Env::_syncEnv()
{
	// Sleep for specified time and Wait for env to respond to changes
	_sleep();

	pthread_mutex_lock(_lock);
	for (int i = 0; i < 2; i++) {
		
		if (_disableServo[i]) {
			continue;
		}

		while (_params->eventData[i].timestamp == _lastTimeStamp[i]) {
			pthread_cond_wait(_cond, _lock);
		}

		_lastData[i] = _currentData[i];
		_currentData[i] = _params->eventData[i];
	}

	pthread_mutex_unlock(_lock);
}

void Env::step(double actions[2][3])
{

	SD currentState[2];

	_lastReward[0] = 0.0;
	_lastReward[1] = 0.0;
	ED _tempData[2];
	SD _tempState[2];


	if (_config->frameSkip) {

		for (int servo = 0; servo < 2; servo++) {
			
			if (_disableServo[servo]) {
				continue;
			}

			// Hold onto origional state for framskip
			_tempData[servo] = _currentData[servo];
			_tempState[servo] = _currentState[servo];

			for (int a = 0; a < _config->numActions; a++) {
				_lastActions[servo][a] = actions[servo][a];
				actions[servo][a] = Utility::rescaleAction(actions[servo][a], _params->config->actionLow, _params->config->actionHigh);
			}

			// Print out the PID gains
			if (0.1 >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
				std::cout << "Here are the gains: ";
				for (int a = 0; a < _params->config->numActions; a++) {
					std::cout << actions[servo][a] << ", ";
				}
				std::cout << std::endl;
			}

			if (_lastData[servo].done) {
				_pids[servo]->init();
			}

			_pids[servo]->setWeights(actions[servo][0], actions[servo][1], actions[servo][2]);
		}

		for (int currentFrameSkip = 0; currentFrameSkip < _config->numFrameSkip; currentFrameSkip++) {
			
			for (int servo = 0; servo < 2; servo++) {

				if (_disableServo[servo]) {
					continue;
				}

				double newAngle = _pids[servo]->update(_currentData[servo].Obj, 1000.0 / static_cast<double>(_params->rate));
				newAngle = Utility::mapOutput(newAngle, _params->config->pidOutputLow, _params->config->pidOutputHigh, _params->config->angleLow, _params->config->angleHigh);
				
				if (_invert[servo]) {
					newAngle = _params->config->angleHigh - newAngle;
				}
				
				_lastAngles[servo] = _currentAngles[servo];
				_currentAngles[servo] = newAngle;

				Utility::runServo(servo, newAngle);
			}

			_syncEnv();
			
			for (int i = 0; i < 2; i++) {

				if (_disableServo[i]) {
					continue;
				}

				double lastError = _lastData[i].error;
				double currentError = _currentData[i].error;

				if (_lastData[i].done) {
					_lastReward[i] =+ Utility::errorToReward(currentError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.02, true) / static_cast<double>(_params->config->numFrameSkip);
				}
				else {
					_lastReward[i] =+ Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.02, true) / static_cast<double>(_params->config->numFrameSkip);
				}
			}

			_updateState();	

			// Ran into a terminal state, no more steps can be taken
			for (int i = 0; i < 2; i++) {

				if (_disableServo[i]) {
					continue;
				}

				if (_currentData[i].done) {
					
					// Set what was the origional current data/state before frameskip
					// to the lastData after frameskip. 
					for (int k = 0; k < 2; k++) {

						_lastData[k] = _tempData[k];
						_lastState[k] = _tempState[k];
					}
					
					return;
				}
			}
		}
	}
	else {
		double randChance = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
		for (int servo = 0; servo < 2; servo++) {

			if (_disableServo[servo]) {
				continue;
			}

			for (int a = 0; a < _config->numActions; a++) {
				_lastActions[servo][a] = actions[servo][a];
				actions[servo][a] = Utility::rescaleAction(actions[servo][a], _params->config->actionLow, _params->config->actionHigh);
			}
			
			// Print out the PID gains
			if (0.01 >= randChance) {
				std::cout << "Here are the gains: ";
				for (int a = 0; a < _params->config->numActions; a++) {
					std::cout << actions[servo][a] << ", ";
				}
				std::cout << std::endl;
			}

			if (_lastData[servo].done) {
				_pids[servo]->init();
			}

			_pids[servo]->setWeights(actions[servo][0], actions[servo][1], actions[servo][2]);

			double newAngle = _pids[servo]->update(_currentData[servo].Obj, 1000.0 / static_cast<double>(_params->rate));
			newAngle = Utility::mapOutput(newAngle, _params->config->pidOutputLow, _params->config->pidOutputHigh, _params->config->angleLow, _params->config->angleHigh);
			_lastAngles[servo] = _currentAngles[servo];
			_currentAngles[servo] = newAngle;

			if (_invert[servo]) {
				newAngle = _params->config->angleHigh - newAngle;
			}

			Utility::runServo(servo, newAngle);
		}

		_syncEnv();

		for (int i = 0; i < 2; i++) {

			if (_disableServo[i]) {
				continue;
			}

			double lastError = _lastData[i].Obj;
			double currentError = _currentData[i].Obj;

			if (_lastData[i].done) {
				_lastReward[i] = Utility::errorToReward(currentError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.02, true) / 2.0;
			}
			else {
				_lastReward[i] = Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.02, true) / 2.0;
			}

			// print out the data tuple
			if (0.01 >= randChance) {
				std::cout << "Here is last Error, current Error, and reward: (" << lastError << ", " << currentError << ", " << _lastReward[i] << ")" << std::endl;
			}
			
		}

		_updateState();

		// Ran into a terminal state, no more steps can be taken
		for (int i = 0; i < 2; i++) {

			if (_disableServo[i]) {
				continue;
			}

			if (_currentData[i].done) {
				return;
			}
		}
	}

	
}

void Env::getResults(TD trainData[2])
{
	for (int i = 0; i < 2; i++) {

		if (_disableServo[i]) {
			continue;
		}

		trainData[i].done = _currentData[i].done;
		trainData[i].reward = _lastReward[i];
		trainData[i].nextState = _currentState[i];
		trainData[i].currentState = _lastState[i];

		for (int a = 0; a < _config->numActions; a++) {
			trainData[i].actions[a] = _lastActions[i][a];
		}
	}
}

bool Env::isDone()
{
	for (int servo = 0; servo < 2; servo++) {
		if (_disableServo[servo]) {
			continue;
		}
		else {
			return _currentData[servo].done;
		}
	} 

	return true;
}

bool Env::hasData()
{
	return _hasData;
}

void Env::resetEnv()
{

	for (int i = 0; i < 2; i++) {
		Utility::runServo(i, _resetAngles[i]);
		_currentAngles[i] = _resetAngles[i];
		_pids[i]->init();
	}
}

void Env::_resetData()
{
	_hasData = false;

	std::memset(_stateData, 0.0, sizeof _stateData);
	std::memset(_lastActions, 0.0, sizeof _lastActions);

	for (int i = 0; i < 2; i++) {
		_errorCounter = -1;

		_currentData[i].reset();
		_lastData[i].reset();	

		_lastAngles[i] = _resetAngles[i];
		_currentAngles[i] = _resetAngles[i];
	}
}



/*
	int num_inputs = 2;
	int num_keys = 7;
	_errorCounter = (_errorCounter + 1) % num_inputs;

	_stateData[i][0][_errorCounter] = data[i].timestamp;
	_stateData[i][1][_errorCounter] = (data[i].Obj - (static_cast<double>(_params->dims[i]) / 2.0));
	_stateData[i][2][_errorCounter] = data[i].Frame / scale;
	_stateData[i][3][_errorCounter] = data[i].Obj / scale;
	_stateData[i][4][_errorCounter] = data[i].point / scale;
	_stateData[i][5][_errorCounter] = data[i].size / scale;
	_stateData[i][6][_errorCounter] = _currentAngles[i];

	int k, j;

	double keys[num_keys];

	// Insertion sort by timestamp
	for (k = 1; k < num_inputs; k++)
	{
		for (int h = 0; h < num_keys; h++) {
			keys[h] = _stateData[i][h][k];
		}

		j = k - 1;

		while (j >= 0 && _stateData[i][0][j] > _stateData[i][0][k])
		{
			for (int h = 0; h < num_keys; h++) {
				_stateData[i][h][j + 1] = _stateData[i][h][j];
			}

			j = j - 1;
		}

		for (int h = 0; h < num_keys; h++) {
			_stateData[i][h][j + 1] = keys[h];
		}

	} 
	*/

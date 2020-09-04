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

	_pids[0] = parameters->pan;
	_pids[1] = parameters->tilt;

	_currentAngles[0] = 0.0;
	_currentAngles[1] = 0.0;

	_lastAngles[0] = 0.0;
	_lastAngles[1] = 0.0;

	_errorCounter = -1;

	_invert[0] = _params->config->invertY;
	_invert[1] = _params->config->invertX;
}

void Env::_updateState(ED data[2])
{

	_errorCounter = (_errorCounter + 1) % 3;

	for (int i = 0; i < 2; i++) {
		if (_disableServo[i]) {
			continue;
		}
		if (data[i].done) {
			resetEnv();
		}
		

		// Normalize and fill out the current state
		double scale = static_cast<double>(_params->dims[i]) / 2.0;
		/*_lastTimeStamp[i] = data[i].timestamp;
		_stateData[i][0][_errorCounter] = data[i].timestamp;
		_stateData[i][1][_errorCounter] = data[i].error / scale;
		_stateData[i][2][_errorCounter] = data[i].Frame / scale;
		_stateData[i][3][_errorCounter] = data[i].Obj / scale;
		_stateData[i][4][_errorCounter] = data[i].point / scale;
		_stateData[i][5][_errorCounter] = data[i].size / scale;
		_stateData[i][6][_errorCounter] = _currentAngles[i] / _params->config->angleHigh;

		int k, j;
		double key;
		double key2;
		double key3;
		double key4;
		double key5;
		double key6;
		double key7;

		for (k = 1; k < 3; k++)
		{
			key = _stateData[i][0][k];
			key2 = _stateData[i][1][k];
			key3 = _stateData[i][2][k];
			key4 = _stateData[i][3][k];
			key5 = _stateData[i][4][k];
			key6 = _stateData[i][5][k];
			key7 = _stateData[i][6][k];

			j = k - 1;

			while (j >= 0 && _stateData[i][0][j] > key)
			{
				_stateData[i][0][j + 1] = _stateData[i][0][j];
				_stateData[i][1][j + 1] = _stateData[i][1][j];
				_stateData[i][2][j + 1] = _stateData[i][2][j];
				_stateData[i][3][j + 1] = _stateData[i][3][j];
				_stateData[i][4][j + 1] = _stateData[i][4][j];
				_stateData[i][5][j + 1] = _stateData[i][5][j];
				_stateData[i][6][j + 1] = _stateData[i][6][j];

				j = j - 1;
			}
			_stateData[i][0][j + 1] = key;
			_stateData[i][1][j + 1] = key2;
			_stateData[i][2][j + 1] = key3;
			_stateData[i][3][j + 1] = key4;
			_stateData[i][4][j + 1] = key5;
			_stateData[i][5][j + 1] = key6;
			_stateData[i][6][j + 1] = key7;
		}*/

		double state[_params->config->numInput];

		// Last two errors
		state[0] = _currentData[i].error / scale;
		state[1] = _lastData[i].error / scale;

		// Send in the last two angles
		state[2] = _currentAngles[i] / _params->config->angleHigh;
		state[3] = _lastAngles[i] / _params->config->angleHigh;

		// PID gains
		double pid[3] = {
			0.0
		};

		// Send in last integral and dirivitive 
		_pids[i]->getPID(pid);
		state[4] = pid[1] / scale;
		state[5] = pid[2] / (scale);

		_lastState[i] = _currentState[i];
		_currentState[i].setStateArray(state);

		/*	
		// Error, e(t) = y′(t) - y(t)
		state[0] = _stateData[i][1][2];

		// First order error, e(t) - e(t−1)
		state[1] = state[0] - _stateData[i][1][1];

		// Second order error, e(t) − 2∗e(t−1) + e(t−2)
		state[2] = state[0] - 2.0 * _stateData[i][1][1] + _stateData[i][1][0];

		state[3] = _stateData[i][2][0];
		state[4] = _stateData[i][2][1];
		state[5] = _stateData[i][2][2];
		state[6] = _stateData[i][3][0];
		state[7] = _stateData[i][3][1];
		state[8] = _stateData[i][3][2];

		// Last three frame/object size, location, and center data
		state[9] = _stateData[i][4][0];
		state[10] = _stateData[i][4][1];
		state[11] = _stateData[i][4][2];
		state[12] = _stateData[i][5][0];
		state[13] = _stateData[i][5][1];
		state[14] = _stateData[i][5][2];


		/*double weights[3] = {
			0.0
		};

		_pids[i]->getWeights(weights);
		state[15] = weights[0];
		state[16] = weights[1];
		state[17] = weights[2];*/
		if (!_lastData[i].done) {
			_hasData = true;
		}
	}
}

void Env::_getCurrentState(SD currentStates[2])
{
	for (int i = 0; i < 2; i++) {
		if (_disableServo[i]) {
			continue;
		}
		currentStates[i] = _currentState[i];
	}
}

void Env::_getLastState(SD lastState[2])
{
	for (int i = 0; i < 2; i++) {
		if (_disableServo[i]) {
			continue;
		}
		lastState[i] = _lastState[i];
	}
}

void Env::_sleep()
{
	int milis = 1000 / _params->rate;
	Utility::msleep(milis);
}

bool Env::init(SD currentState[2])
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

	_updateState(_currentData);
	_getCurrentState(currentState);

	return true;
}

void Env::ping(SD currentState[2])
{
	_resetData();
	_syncEnv();
	_updateState(_currentData);
	_getCurrentState(currentState);
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

void Env::step(double actions[2][3], SD currentState[2])
{

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
			if (0.01 >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
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

				double newAngle = _pids[servo]->update(_currentData[servo].error, 0);
				// double newAngle = _pids[servo]->update(_currentData[servo].error / (static_cast<double>(_params->dims[servo] / 2.0)), 0);
				// double newAngle = actions[servo][0];
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
					_lastReward[i] =+ Utility::errorToReward(currentError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.01, true);
				}
				else {
					_lastReward[i] =+ Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.01, true);
				}
			}

			_updateState(_currentData);	

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
					
					goto done;
				}
			}
		}
	}
	else {
		for (int servo = 0; servo < 2; servo++) {

			if (_disableServo[servo]) {
				continue;
			}

			for (int a = 0; a < _config->numActions; a++) {
				_lastActions[servo][a] = actions[servo][a];
				actions[servo][a] = Utility::rescaleAction(actions[servo][a], _params->config->actionLow, _params->config->actionHigh);
			}
			
			// Print out the PID gains
			if (0.01 >= static_cast<float>(rand()) / static_cast <float> (RAND_MAX)) {
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

			double newAngle = _pids[servo]->update(_currentData[servo].error, 0);
			// double newAngle = _pids[servo]->update(_currentData[servo].error / (static_cast<double>(_params->dims[servo]) / 2.0), 0);
			// double newAngle = actions[servo][0];
			newAngle = Utility::mapOutput(newAngle, _params->config->pidOutputLow, _params->config->pidOutputHigh, _params->config->angleLow, _params->config->angleHigh);
			_lastAngles[servo] = _currentAngles[servo];
			_currentAngles[servo] = newAngle;

			if (_invert[servo]) {
				newAngle = _params->config->angleHigh - newAngle;
			}

			std::cout << "For servo: " << servo << ", the angle is: " << newAngle << "." << std::endl;
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
				_lastReward[i] = Utility::errorToReward(currentError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.01, true);
			}
			else {
				_lastReward[i] = Utility::pidErrorToReward(currentError, lastError, static_cast<double>(_params->dims[i]) / 2.0, _currentData[i].done, 0.01, true);
			}
		}

		_updateState(_currentData);

		// Ran into a terminal state, no more steps can be taken
		for (int i = 0; i < 2; i++) {

			if (_disableServo[i]) {
				continue;
			}

			if (_currentData[i].done) {
				goto done;
			}
		}
	}

	done:
	_getCurrentState(currentState);
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

	for (int i = 0; i < 2; i++) {
		_errorCounter = -1;
		std::memset(_stateData, 0.0, sizeof _stateData);

		_currentData[i].reset();
		_lastData[i].reset();
		std::memset(_lastActions, 0.0, sizeof _lastActions);

		_lastAngles[i] = _resetAngles[i];
		_currentAngles[i] = _resetAngles[i];
	}
}
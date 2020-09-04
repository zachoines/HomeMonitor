#pragma once
#include "data.h"
class Env
{
private:
	pthread_mutex_t* _lock;
	pthread_cond_t* _cond;
	int _frameSkip;
	param* _params;
	cfg* _config;

	SD _currentState[2];
	SD _lastState[2];
	ED _lastData[2];
	ED _currentData[2];
	
	double _lastReward[2]{
		0.0
	};

	double _stateData[2][8][3] = {
		0.0
	};

	double _lastTimeStamp[2]{
		0.0
	};

	double _lastActions[2][3] = {
		0.0
	}; 

	bool _hasData;
	bool _invert[2];
	bool _disableServo[2];
	double _resetAngles[2];
	double _currentAngles[2];
	double _lastAngles[2];

	PID* _pan = nullptr;
	PID* _tilt = nullptr;
	PID* _pids[2];

	int _errorCounter;

	void _updateState(ED data[2]);
	void _getCurrentState(SD currentStates[2]);
	void _getLastState(SD currentState[2]);
	void _sleep();
	void _syncEnv();
	void _resetData();
public:
	Env(param* parameters, pthread_mutex_t* stateDataLoc, pthread_cond_t* stateDataCond);
	void resetEnv(); // Resets servos and re-inits PID's. Call only once manually.
	bool init(SD currentState[2]); // Call once at beginning of program. Resets internal state, returns current env state. 
	void ping(SD currentState[2]); // Sync with dection results
	void step(double actions[2][3], SD currentState[2]);  // Steps with actions and waits for Env to update, then returns the current state.
	void getResults(TD trainData[2]); // Call after stepping for temporal transition results from env. 
	bool isDone();
	bool hasData();
};


#pragma once
#include "data.h"
#include "PID.h"
class Env
{
private:
	pthread_mutex_t* _lock;
	pthread_cond_t* _cond;
	int _frameSkip;
	param* _params;
	cfg* _config;

	double _lastTimeStamp[NUM_SERVOS];

	ED _lastData[NUM_SERVOS];
	ED _currentData[NUM_SERVOS];
	SD _observation[NUM_SERVOS];

	bool _invert[NUM_SERVOS];
	bool _disableServo[NUM_SERVOS];
	double _resetAngles[NUM_SERVOS];
	double _currentAngles[NUM_SERVOS];
	double _lastAngles[NUM_SERVOS];

	PID* _pan = nullptr;
	PID* _tilt = nullptr;
	PID* _pids[NUM_SERVOS];

	void _sleep();
	void _syncEnv();
	void _resetEnv(); // Resets servos and re-inits PID's. Call only once manually.

public:
	Env(param* parameters, pthread_mutex_t* stateDataLoc, pthread_cond_t* stateDataCond);

	bool isDone();

	RD reset();  // Steps with actions and waits for Env to update, then returns the current state.
	SR step(double actions[NUM_SERVOS][NUM_ACTIONS], bool rescale = true);  // Steps with actions and waits for Env to update, then returns the current state.
};


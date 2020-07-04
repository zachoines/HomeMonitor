#include "SACAgent.h"
// build a neural network similar to how you would do it with Pytorch
/*
		INPUT: [u(n); u(n − 1); y(n); y(n − 1); C(n); C(n − 1); e(n)]
			u(n) and u(n − 1) are reference inputs (desired trajectory)
			y(n) and y(n − 1) are reference outputs (real trajectory)
			C(n) and C(n − 1) are the control signals
			e(n) is the position tracking error, input to PID
		OUTPUT: [Kp; Ki; Kd]
			These are the input gains of the PID


		Error:
			Sum(MSE(K_n - K_n-1))
			Are goal is to minimize the gradient of our MSE error function with respect to our PID input gains. We want
			to converge on stable gains given our current environment at each timestep.

*/
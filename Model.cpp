#include "Model.h"
#include <torch/torch.h>

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
Model::Model()
{
	// construct and register your layers
	in = register_module("in", torch::nn::Linear(7, 7));
	h = register_module("h", torch::nn::Linear(7, 3));
	out = register_module("out", torch::nn::Linear(3, 3));

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

Model::~Model()
{

	// TODO:: Make sure params are saved to model file
	delete optimizer;
}

torch::Tensor Model::forward(torch::Tensor X, torch::Tensor k)
{

	// let's pass relu 
	X = torch::relu(in->forward(X));
	X = torch::relu(h->forward(X));
	X = torch::sigmoid(out->forward(X));


	// this->batches.push_back(X);

	// return the output
	return X;

}

void Model::save()
{
}

void Model::load()
{
}

void Model::train()
{
	// Initialize running metrics
	double running_loss = 0.0;
	size_t num_correct = 0;

	// Collect enough data
	//for (auto& batch : batches) {


	//	optimizer->zero_grad();

	//	torch::Tensor input;
	//	torch::Tensor target;


	//	// Calculate loss with error function and determine gradients for backward pass
	//	auto loss = torch::mse_loss(input, target.detach());

	//	// Update the weights	
	//	optimizer->step();
	//}
}
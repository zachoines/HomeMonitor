#pragma once
#include <torch/torch.h>

struct QNetwork : torch::nn::Module {

private:
	float_t learning_rate;
	int num_inputs, num_actions, hidden_size, init_w;
	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, linear3{ nullptr };

public:
	torch::optim::Adam* optimizer = nullptr;

	// Constructor
	QNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int learning_rate = 1e-4);
	~QNetwork();
	torch::Tensor forward(torch::Tensor state, torch::Tensor actions);

} typedef QN;

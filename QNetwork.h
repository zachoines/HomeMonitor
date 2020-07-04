#pragma once
#include <torch/torch.h>

struct QNetwork : torch::nn::Module {

private:
	float_t learning_rate;
	torch::optim::Adam* optimizer = nullptr;
	int num_inputs, num_actions, hidden_size, init_w;

	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, linear3{ nullptr };

public:
	// Constructor
	QNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int learning_rate = 1e-4);
	QNetwork();
	~QNetwork();

	torch::Tensor forward(torch::Tensor state);

	void save(std::stringstream stream);
	void load();
	void train();

};
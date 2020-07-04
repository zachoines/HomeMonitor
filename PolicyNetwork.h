#pragma once
#include <torch/torch.h>
struct PolicyNetwork: torch::nn::Module 
{
private:
	int num_inputs, num_actions, hidden_size, init_w, log_std_min, log_std_max;
	
	float_t learning_rate = 1e-4;
	torch::optim::Adam* optimizer = nullptr;
	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, mean_Linear{ nullptr }, log_std_linear{ nullptr };
public:
	PolicyNetwork(int num_inputs, int num_actions, int hidden_size, int init_w, int log_std_min, int log_std_max);
	~PolicyNetwork();
	at::TensorList forward(torch::Tensor state);
	at::TensorList sample(torch::Tensor state, double epsilon = 1e-6);
} typedef PN;


#pragma once
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/nn.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>
#include <utility>

struct PolicyNetwork: torch::nn::Module 
{
private:
	int num_inputs, num_actions, hidden_size, init_w, log_std_min, log_std_max;
	float_t learning_rate = 1e-4;
	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, mean_Linear{ nullptr }, log_std_linear{ nullptr };

public:
	torch::optim::Adam* optimizer = nullptr;
	PolicyNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int log_std_min = -20, int log_std_max = 2);
	~PolicyNetwork();
	at::TensorList forward(torch::Tensor state);
	at::TensorList sample(torch::Tensor state, double epsilon = 1e-6);

} typedef PN;


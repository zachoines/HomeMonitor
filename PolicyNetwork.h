#pragma once
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/nn.h>
#include <torch/serialize/archive.h>
#include <torch/serialize/tensor.h>
#include <utility>

struct PolicyNetwork: torch::nn::Module 
{
private:
	int num_inputs, num_actions, hidden_size, log_std_min, log_std_max;
	double learning_rate, init_w;
	torch::nn::Linear linear1{ nullptr }, linear2{ nullptr }, mean_Linear{ nullptr }, log_std_linear{ nullptr };

public:
	torch::optim::Adam* optimizer = nullptr;
	PolicyNetwork(int num_inputs, int num_actions, int hidden_size, double init_w = 3e-4, int log_std_min = -20, int log_std_max = 2);
	~PolicyNetwork();
	at::Tensor forward(torch::Tensor state);
	at::Tensor sample(torch::Tensor state, int batchSize, double epsilon = 1e-6);

} typedef PN;


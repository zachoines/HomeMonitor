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
	PolicyNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int log_std_min = -20, int log_std_max = 2);
	~PolicyNetwork();
	at::TensorList forward(torch::Tensor state);
	at::TensorList sample(torch::Tensor state, double epsilon = 1e-6);

	void save_to(std::stringstream& stream);
	void load_from(std::stringstream& stream);
	void save_to(const std::string& file_name);
	void load_from(const std::string& file_name);

} typedef PN;


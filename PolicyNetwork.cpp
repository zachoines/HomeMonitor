#include "PolicyNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <regex>
#include <stack>
#include <torch/torch.h>
#include "Normal.h"

PolicyNetwork::PolicyNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int log_std_min = -20, int log_std_max = 2) {
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->log_std_min = log_std_min;
	this->log_std_max = log_std_max;

	// Set network structure
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	mean_Linear = register_module("mean_Linear", torch::nn::Linear(hidden_size, num_actions));
	log_std_linear = register_module("log_std_linear", torch::nn::Linear(hidden_size, num_actions));

	// Initialize params
	auto p = this->named_parameters(false);
	auto weights = p.find("weight");
	auto biases = p.find("bias");

	if (weights != nullptr) torch::nn::init::uniform_(*weights);
	if (biases != nullptr) torch::nn::init::uniform_(*biases);
}

at::TensorList PolicyNetwork::forward(torch::Tensor state) {
	torch::Tensor X, mean, log_std;
	
	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	mean = mean_Linear->forward(X);
	log_std = log_std_linear->forward(X);
	log_std = torch::clamp(log_std, log_std_min, log_std_max);

	return { mean, log_std };
}

at::TensorList PolicyNetwork::sample(torch::Tensor state, double epsilon = 1e-6) {
	torch::Tensor X, mean, log_std, std, z, action, log_prob, log_pi;

	at::TensorList result = this->forward(state);
	mean = result[0];
	log_std = result[1];
	std = torch::exp(log_std);
	Normal::Normal normal = Normal(mean, std);
	z = normal.rsample({num_actions});
	action = torch::tanh(z);

	log_pi = normal.log_prob(z) - torch::log(1 - action.pow(2) + epsilon);
	log_pi = log_pi.sum(1, true);

	return { action, log_pi };
}

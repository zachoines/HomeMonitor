#include "PolicyNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <regex>
#include <stack>
#include <torch/torch.h>
#include "Normal.h"

PolicyNetwork::PolicyNetwork(int num_inputs, int num_actions, int hidden_size, double init_w, int log_std_min, int log_std_max, double learning_rate) {
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->log_std_min = log_std_min;
	this->log_std_max = log_std_max;
	this->learning_rate = learning_rate;

	// Set network structure
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	mean_Linear = register_module("mean_Linear", torch::nn::Linear(hidden_size, num_actions));
	log_std_linear = register_module("log_std_linear", torch::nn::Linear(hidden_size, num_actions));

	// Initialize params
	torch::autograd::GradMode::set_enabled(false);
	
	linear1->weight.uniform_(-init_w, init_w);
	linear2->weight.uniform_(-init_w, init_w);
	mean_Linear->weight.uniform_(-init_w, init_w);
	log_std_linear->weight.uniform_(-init_w, init_w);
	linear1->bias.uniform_(-init_w, init_w);
	linear2->bias.uniform_(-init_w, init_w);
	mean_Linear->bias.uniform_(-init_w, init_w);
	log_std_linear->bias.uniform_(-init_w, init_w);

	torch::autograd::GradMode::set_enabled(true);

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

PolicyNetwork::~PolicyNetwork() {
	delete optimizer;
}

torch::Tensor PolicyNetwork::forward(torch::Tensor state) {
	torch::Tensor X, mean, log_std;

	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	mean = mean_Linear->forward(X);
	log_std = log_std_linear->forward(X);
	log_std = torch::clamp(log_std, log_std_min, log_std_max);

	return torch::cat({ { mean }, { log_std } }, 0);
}

torch::Tensor PolicyNetwork::sample(torch::Tensor state, int batchSize, double epsilon) {
	torch::Tensor X, mean, log_std, std, z, action, log_prob, log_pi;
	
	at::Tensor result = this->forward(state);
	at::Tensor reshapedResult = result.view({2, batchSize, num_actions });
	
	mean = reshapedResult[0];
	log_std = reshapedResult[1];
	std = torch::exp(log_std);
	Normal normal = Normal(mean, std);
	z = normal.rsample();
	action = torch::tanh(z);
	log_pi = normal.log_prob(z) - torch::log(1 - torch::add(torch::pow(action, 2), epsilon));

	return torch::cat({ { action }, {log_pi}, { mean }, { std }, { z }}, 0);
}

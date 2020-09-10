#include "PolicyNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <regex>
#include <stack>
#include <cmath>
#include <torch/torch.h>
#include "Normal.h"

PolicyNetwork::PolicyNetwork(int num_inputs, int num_actions, int hidden_size, double init_w, int log_std_min, int log_std_max, double learning_rate, double action_max, double action_min) {
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->log_std_min = log_std_min;
	this->log_std_max = log_std_max;
	this->learning_rate = learning_rate;
	_action_max = action_max;
	_action_min = action_min;
	_action_scale = (action_max - action_min) / 2.0;
	_action_bias = (action_max + action_min) / 2.0;

	// Set network structure
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	// dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions().p(0.5)));
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

	linear1->weight.set_requires_grad(true);
	linear2->weight.set_requires_grad(true);
	mean_Linear->weight.set_requires_grad(true);
	log_std_linear->weight.set_requires_grad(true);
	linear1->bias.set_requires_grad(true);
	linear2->bias.set_requires_grad(true);
	mean_Linear->bias.set_requires_grad(true);
	log_std_linear->bias.set_requires_grad(true);

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

PolicyNetwork::~PolicyNetwork() {
	delete optimizer;
}

torch::Tensor PolicyNetwork::forward(torch::Tensor state, bool eval) {
	torch::Tensor X, mean, log_std;
	 
	//if (eval) {
	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	/*}
	else {
		X = dropout->forward(torch::relu(linear1->forward(state)));
		X = dropout->forward(torch::relu(linear2->forward(X)));
	}*/
	
	
	// mean = torch::tanh(mean_Linear->forward(X));
	// log_std = torch::tanh(log_std_linear->forward(X));
	// log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0);
	
	mean = mean_Linear->forward(X);
	log_std = log_std_linear->forward(X);
	log_std = torch::clamp(log_std, log_std_min, log_std_max);

	return torch::cat({ { mean }, { log_std } }, 0);
}


torch::Tensor PolicyNetwork::sample(torch::Tensor state, int batchSize, double epsilon, bool eval) {

	at::Tensor result = this->forward(state, eval);
	at::Tensor reshapedResult = result.view({ 2, batchSize, num_actions });

	torch::Tensor mean = reshapedResult[0];
	torch::Tensor log_std = reshapedResult[1];
	torch::Tensor std = torch::exp(log_std);
                                                                                                                  
	Normal normal = Normal(mean, std); 
	torch::Tensor z = normal.rsample(); // Reparameterization
	torch::Tensor action = torch::tanh(z);
	torch::Tensor log_probs = normal.log_prob(z);

	// Rescale to action bounds
	torch::Tensor action_scaled = action * _action_scale + _action_bias;
	torch::Tensor log_probs_scaled = log_probs - torch::log(_action_scale * (1.0 - action.pow(2)) + epsilon);
	torch::Tensor mean_scaled = torch::tanh(mean) * _action_scale + _action_bias;
	
	// log_probs = log_probs - torch::log(1.0 - action.pow(2) + epsilon);

	return torch::cat({ { action_scaled }, {log_probs_scaled}, { mean_scaled}, { std }, { z } }, 0);

}

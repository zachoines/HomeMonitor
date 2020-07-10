#include "QNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/torch.h>


QNetwork::QNetwork(int num_inputs, int num_actions, int hidden_size, int init_w, int learning_rate)
{
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->learning_rate = learning_rate;

	// construct and register your layers
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs + num_actions, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	linear3 = register_module("linear3", torch::nn::Linear(hidden_size, num_actions));

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));

	auto p = this->named_parameters(false);
	auto weights = p.find("weight");
	auto biases = p.find("bias");

	if (weights != nullptr) torch::nn::init::uniform_(*weights);
	if (biases != nullptr) torch::nn::init::uniform_(*biases);
}

QNetwork::~QNetwork()
{
	delete optimizer;
}

torch::Tensor QNetwork::forward(torch::Tensor state, torch::Tensor actions)
{
	torch::Tensor X;

	X = torch::relu(linear1->forward(torch::cat({ state, actions }, 1)));
	X = torch::relu(linear2->forward(X));
	X = linear3->forward(X);

	return X;

}






#include "QNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/torch.h>


QNetwork::QNetwork(int num_inputs, int num_actions, int hidden_size, int init_w = 3e-3, int learning_rate = 1e-4)
{
	this->num_inputs = num_inputs;
	this->num_actions = num_actions;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->learning_rate = learning_rate;

	// construct and register your layers
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
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

	// TODO:: Make sure params are saved to QNetwork file
	delete optimizer;
}

torch::Tensor QNetwork::forward(torch::Tensor state)
{
	torch::Tensor X;

	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	X = linear3->forward(X);

	return X;

}

void QNetwork::save_to(std::stringstream& stream)
{
	 torch::save(this, stream);
}

void QNetwork::load_from(std::stringstream& stream)
{
	torch::load(this, stream);
}

void QNetwork::save_to(const std::string& file_name) {
	torch::save(this, file_name);
}

void QNetwork::load_from(const std::string& file_name) {
	torch::load(this, file_name);
}





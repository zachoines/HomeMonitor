#include "QNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <regex>
#include <stack>
#include <torch/torch.h>


QNetwork::QNetwork(int num_inputs, int num_actions, int hidden_size, int init_w, int learning_rate)
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
QNetwork::QNetwork()
{
	num_inputs = 7;
	num_actions = 3;
	hidden_size = 7;
	init_w = 3e-3;
	learning_rate = 1e-4;

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

void QNetwork::save(std::stringstream stream)
{

	// auto params = this->named_parameters(true /*recurse*);
	// auto buffers = this->named_buffers(true /*recurse*/);
	// torch::save(this, stream);

}

void QNetwork::load()
{
}

void QNetwork::train()
{
	// Initialize running metrics
	double running_loss = 0.0;
	size_t num_correct = 0;

	// Collect enough data
	//for (auto& batch : batches) {


	//	optimizer->zero_grad();

	//	torch::Tensor input;
	//	torch::Tensor target;


	//	// Calculate loss with error function and determine gradients for backward pass
	//	auto loss = torch::mse_loss(input, target.detach());

	//	// Update the weights	
	//	optimizer->step();
	//}
}

//
//void SaveStateDict(const torch::nn::Module& module,
//	const std::string& file_name) {
//	torch::serialize::OutputArchive archive;
//	auto params = module.named_parameters(true /*recurse*/);
//	auto buffers = module.named_buffers(true /*recurse*/);
//	for (const auto& val : params) {
//		if (!torch::is_empty(val.value())) {
//			archive.write(val.key(), val.value());
//		}
//	}
//	for (const auto& val : buffers) {
//		if (!is_empty(val.value())) {
//			archive.write(val.key(), val.value(), /*is_buffer*/ true);
//		}
//	}
//	archive.save_to(file_name);
//}
//
//void LoadStateDict(torch::nn::Module& module,
//	const std::string& file_name,
//	const std::string& ignore_name_regex) {
//	torch::serialize::InputArchive archive;
//	archive.load_from(file_name);
//	torch::NoGradGuard no_grad;
//	std::regex re(ignore_name_regex);
//	std::smatch m;
//	auto params = module.named_parameters(true /*recurse*/);
//	auto buffers = module.named_buffers(true /*recurse*/);
//	for (auto& val : params) {
//		if (!std::regex_match(val.key(), m, re)) {
//			archive.read(val.key(), val.value());
//		}
//	}
//	for (auto& val : buffers) {
//		if (!std::regex_match(val.key(), m, re)) {
//			archive.read(val.key(), val.value(), /*is_buffer*/ true);
//		}
//	}
//}


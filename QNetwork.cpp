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

	torch::autograd::GradMode::set_enabled(false);
	
	linear1->weight.uniform_(-init_w, init_w);
	linear2->weight.uniform_(-init_w, init_w);
	linear3->weight.uniform_(-init_w, init_w);
	linear1->bias.uniform_(-init_w, init_w);
	linear2->bias.uniform_(-init_w, init_w);
	linear3->bias.uniform_(-init_w, init_w);

	/*torch::nn::init::kaiming_normal_(linear1->weight);
	torch::nn::init::kaiming_normal_(linear2->weight);
	torch::nn::init::kaiming_normal_(linear3->weight);
	linear1->bias.zero_();
	linear2->bias.zero_();
	linear3->bias.zero_();*/

	// auto p = this->named_parameters(true);
	//for (auto& val : p) {
	//	std::cout << "Key: " << val.key() << ", Value: " << val.value() << std::endl;
	//}

	torch::autograd::GradMode::set_enabled(true);
	
	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
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






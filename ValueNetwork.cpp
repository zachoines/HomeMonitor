#include "ValueNetwork.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/torch.h>


ValueNetwork::ValueNetwork(int num_inputs, int hidden_size, double init_w, double learning_rate)
{
	this->num_inputs = num_inputs;
	this->hidden_size = hidden_size;
	this->init_w = init_w;
	this->learning_rate = learning_rate;

	// construct and register your layers
	linear1 = register_module("linear1", torch::nn::Linear(num_inputs, hidden_size));
	linear2 = register_module("linear2", torch::nn::Linear(hidden_size, hidden_size));
	linear3 = register_module("linear3", torch::nn::Linear(hidden_size, 1));
	// dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions().p(0.5)));

	torch::autograd::GradMode::set_enabled(false);

	linear1->weight.uniform_(-init_w, init_w);
	linear2->weight.uniform_(-init_w, init_w);
	linear3->weight.uniform_(-init_w, init_w);
	linear1->bias.uniform_(-init_w, init_w);
	linear2->bias.uniform_(-init_w, init_w);
	linear3->bias.uniform_(-init_w, init_w);

	torch::autograd::GradMode::set_enabled(true);

	linear1->weight.set_requires_grad(true);
	linear2->weight.set_requires_grad(true);
	linear3->weight.set_requires_grad(true);
	linear1->bias.set_requires_grad(true);
	linear2->bias.set_requires_grad(true);
	linear3->bias.set_requires_grad(true);

	optimizer = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate));
}

ValueNetwork::~ValueNetwork()
{
	delete optimizer;
}

torch::Tensor ValueNetwork::forward(torch::Tensor state, bool eval)
{
	torch::Tensor X;

	/*X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	X = torch::relu(dropout->forward(X));
	X = linear3->forward(X);*/

	// if (eval) {
	X = torch::relu(linear1->forward(state));
	X = torch::relu(linear2->forward(X));
	X = linear3->forward(X);
	/*}
	else {
		X = dropout->forward(torch::relu(linear1->forward(state)));
		X = dropout->forward(torch::relu(linear2->forward(X)));
		X = linear3->forward(X);
	}*/

	return X;

}






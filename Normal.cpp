#include "Normal.h"
#include <cmath>
#define _USE_MATH_DEFINES

Normal::Normal(double loc, double scale)
{
	_loc = loc;
	_scale = scale;
	
	this->loc = torch::tensor(_loc);
	this->scale = torch::tensor(_scale);
}

Normal::Normal(torch::Tensor loc, torch::Tensor scale)
{
	this->loc = loc;
	this->scale = scale;

	_loc = loc.item<double>();
	_scale = scale.item<double>();
}

Normal::Normal()
{
	_loc = 0.0;
	_scale = 1.0;
	this->loc = torch::tensor(_loc);
	this->scale = torch::tensor(_scale);
}

torch::Tensor Normal::rsample(c10::IntArrayRef sample_shape)
{	
	return torch::normal(this->_loc, this->_scale, sample_shape);
}

torch::Tensor Normal::log_prob(torch::Tensor value)
{

	at::TensorList tensors = torch::broadcast_tensors({ loc, scale });
	loc = tensors[0];
	scale = tensors[1];

	torch::Tensor var = scale.pow(2);
	torch::Tensor log_scale = var.log();

	return -((value - loc).pow(2) / (2 * var) - log_scale - torch::log(torch::sqrt(torch::tensor(2 * M_PI))));
}

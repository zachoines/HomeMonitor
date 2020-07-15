#include "Normal.h"
#include <cmath>
#define _USE_MATH_DEFINES


Normal::Normal(torch::Tensor loc, torch::Tensor scale)
{
	this->loc = loc;
	this->scale = scale;
}


torch::Tensor Normal::sample()
{	
	return at::normal(loc, scale);
}

torch::Tensor Normal::rsample() {
	torch::Tensor eps = loc.clone();
	eps.zero_().normal_();
	return loc + eps * scale;
}

torch::Tensor Normal::log_prob(torch::Tensor value)
{

	at::TensorList tensors = torch::broadcast_tensors({ loc, scale, value });
	loc = tensors[0];
	scale = tensors[1];
	value = tensors[2];

	torch::Tensor var = scale.pow(2);
	torch::Tensor log_scale = var.log();

	return -((value - loc).pow(2) / (2 * var) - log_scale - torch::log(torch::sqrt(torch::tensor(2 * M_PI))));
}

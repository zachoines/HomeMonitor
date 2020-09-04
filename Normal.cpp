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
	torch::autograd::GradMode::set_enabled(false);
	return at::normal(loc, scale);
}

torch::Tensor Normal::rsample() {
	torch::Tensor eps = torch::empty(loc.data().sizes());
	torch::Tensor normal = loc + eps.normal_() * scale;
	// std::cout << normal << std::endl;
	return normal;
}

torch::Tensor Normal::log_prob(torch::Tensor value)
{

	torch::Tensor var = scale.pow(2);
	torch::Tensor log_scale = var.log();
	
	at::Tensor log_prob = -((value - loc).pow(2)) / (2.0 * var) - log_scale - std::log(std::sqrt(2.0 * M_PI));
	
	return log_prob;

	// return -((value - loc).pow(2.0) / (2.0 * var) - log_scale - torch::log(torch::sqrt(torch::tensor({ 2.0 * M_PI }))));
}

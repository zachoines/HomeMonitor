#pragma once
#include <torch/torch.h>
#include <cmath>
class Normal
{
private:

	torch::Tensor loc;
	torch::Tensor scale;

public:
	Normal(torch::Tensor loc, torch::Tensor scale);

	torch::Tensor sample();
	torch::Tensor rsample();
	torch::Tensor log_prob(torch::Tensor value);
};


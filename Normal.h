#pragma once
#include <torch/torch.h>
#include <cmath>
class Normal
{
private:
	double _loc;
	double _scale;
	torch::Tensor loc;
	torch::Tensor scale;

public:
	Normal(double loc, double scale);
	Normal(torch::Tensor loc, torch::Tensor scale);
	Normal(); // Default loc = 0.0 and scale = 1.0

	torch::Tensor rsample(c10::IntArrayRef sample_shape);
	torch::Tensor log_prob(torch::Tensor value);
};


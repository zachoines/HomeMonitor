#pragma once
#include <torch/torch.h>

struct Model: torch::nn::Module{

private:
	float_t learning_rate = 1e-4;
	int batch_size = 100;
	torch::optim::Adam* optimizer = nullptr;
	torch::nn::Linear in{ nullptr }, h{ nullptr }, out{ nullptr };

public:
	// Constructor
	Model();
	~Model();

	torch::Tensor forward(torch::Tensor X, torch::Tensor k);

	void save();
	void load();
	void train();

} typedef PIDAutoTuner;

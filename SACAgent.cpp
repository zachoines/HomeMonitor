#include "SACAgent.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "Normal.h"
#include "util.h"
#include "data.h"

// build a neural network similar to how you would do it with Pytorch
/*
		INPUT: [u(n); u(n − 1); y(n); y(n − 1); C(n); C(n − 1); e(n)]
			u(n) and u(n − 1) are reference inputs (desired trajectory)
			y(n) and y(n − 1) are reference outputs (real trajectory)
			C(n) and C(n − 1) are the control signals
			e(n) is the position tracking error, input to PID
		OUTPUT: [Kp; Ki; Kd]
			These are the input gains of the PID


		Error:
			Sum(MSE(K_n - K_n-1))
			Are goal is to minimize the gradient of our MSE error function with respect to our PID input gains. We want
			to converge on stable gains given our current environment at each timestep.

*/

SACAgent::SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, double gamma, double tau, double alpha, double q_lr, double p_lr, double a_lr)
{
	_num_inputs = num_inputs;
	_num_actions = num_actions;
	_action_max = action_max;
	_action_min = action_min;

	_gamma = gamma;
	_tau = tau;
	_alpha = alpha;
	_a_lr = a_lr;
	_q_lr = q_lr;
	_p_lr = p_lr;

	// initialize networks
	_q_net1 = new QNetwork(num_inputs, num_actions, num_hidden);
	_q_net2 = new QNetwork(num_inputs, num_actions, num_hidden);
	_target_q_net1 = new QNetwork(num_inputs, num_actions, num_hidden);
	_target_q_net2 = new QNetwork(num_inputs, num_actions, num_hidden);
	_policy_net = new PolicyNetwork(num_inputs, num_actions, num_hidden);

	// Load last checkpoint if available
	load_checkpoint();
	
	// Copy over params
	std::stringstream stream1;
	save_to(*_q_net1, stream1);
	load_from(*_target_q_net1, stream1);

	std::stringstream stream2;
	save_to(*_q_net2, stream2);
	load_from(*_target_q_net2, stream2);

	// Auto Entropy adjustment variables
	_target_entropy = c10::Scalar(-1 * num_actions);
	_log_alpha = torch::zeros(1);
	_log_alpha.set_requires_grad(true);
	_alpha_optimizer = new torch::optim::Adam({_log_alpha}, torch::optim::AdamOptions(a_lr));

}

SACAgent::~SACAgent() {

	delete _q_net1;
	delete _q_net2;
	delete _target_q_net1;
	delete _target_q_net2;
	delete _policy_net;
	delete _alpha_optimizer;
}

void SACAgent::save_to(torch::nn::Module& module, std::stringstream& fd) {
	torch::serialize::OutputArchive archive;
	auto params = module.named_parameters(true /*recurse*/);
	auto buffers = module.named_buffers(true /*recurse*/);
	for (const auto& val : params) {
		archive.write(val.key(), val.value());
	}
	for (const auto& val : buffers) {
		archive.write(val.key(), val.value(), /*is_buffer*/ true);
	}

	archive.save_to(fd);
}

void SACAgent::load_from(torch::nn::Module& module, std::stringstream& fd) {
	torch::serialize::InputArchive archive;
	archive.load_from(fd);
	torch::NoGradGuard no_grad;
	std::smatch m;
	auto params = module.named_parameters(true);
	auto buffers = module.named_buffers(true);
	for (auto& val : params) {
		archive.read(val.key(), val.value());
	}
	for (auto& val : buffers) {
		archive.read(val.key(), val.value(), true);
	}
}

void SACAgent::save_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = "/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = "/Q_Net_Checkpoint2.pt";
	std::string PModelFile = "/P_Net_Checkpoint.pt";

	torch::serialize::OutputArchive QModelArchive1;
	QModelArchive1.save_to(QModelFile1);
	_q_net1->save(QModelArchive1);

	torch::serialize::OutputArchive QModelArchive2;
	QModelArchive2.save_to(QModelFile2);
	_q_net2->save(QModelArchive2);

	torch::serialize::OutputArchive PModelArchive1;
	PModelArchive1.save_to(PModelFile);
	_policy_net->save(PModelArchive1);
}

void SACAgent::load_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = "/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = "/Q_Net_Checkpoint2.pt";
	std::string PModelFile = "/P_Net_Checkpoint.pt";

	if (Utility::fileExists(QModelFile1) && Utility::fileExists(QModelFile2) && Utility::fileExists(PModelFile)) {
		torch::serialize::InputArchive QModelArchive1;
		QModelArchive1.load_from(QModelFile1);
		_q_net1->load(QModelArchive1);

		torch::serialize::InputArchive QModelArchive2;
		QModelArchive2.load_from(QModelFile2);
		_q_net2->load(QModelArchive2);

		torch::serialize::InputArchive PModelArchive1;
		PModelArchive1.load_from(PModelFile);
		_policy_net->load(PModelArchive1);
	}
}

torch::Tensor SACAgent::get_action(torch::Tensor state)
{

	at::TensorList results = _policy_net->forward(state);
	at::Tensor mean = results[0];
	at::Tensor log_std = results[1];
	at::Tensor std = log_std.exp();

	Normal normal = Normal(mean, std);
	torch::Tensor z = normal.sample();
	z.set_requires_grad(false);
	torch::Tensor actions = torch::tanh(z);
	
	// Normalize actions
	actions = _action_min + (actions + 1.0) * 0.5 * (_action_max - _action_min);
	actions.clamp(_action_min, _action_max);

	return actions;
}

void SACAgent::update(int batchSize, Buffer* replayBuffer)
{

}

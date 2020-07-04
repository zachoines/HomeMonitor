#include "SACAgent.h"
#include <bits/stdc++.h> 
#include <iostream>
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

SACAgent::SACAgent(int num_inputs, int num_actions, double action_max, double action_min, double gamma = 0.99, double tau = 0.01, double alpha = 0.2, double q_lr = 3e-4, double p_lr = 3e-4, double a_lr = 3e-4)
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
	_q_net1 = new QNetwork::QNetwork(num_inputs, num_actions, 7);
	_q_net2 = new QNetwork::QNetwork(num_inputs, num_actions, 7);
	_target_q_net1 = new QNetwork::QNetwork(num_inputs, num_actions, 7);
	_target_q_net2 = new QNetwork::QNetwork(num_inputs, num_actions, 7);
	_policy_net = new PolicyNetwork::PolicyNetwork(num_inputs, num_actions, 7);

	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = "/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = "/Q_Net_Checkpoint2.pt";
	std::string PModelFile = "/P_Net_Checkpoint.pt";

	if (Utility::fileExists(QModelFile1) && Utility::fileExists(QModelFile2) && Utility::fileExists(PModelFile)) {
		_q_net1->load_from(QModelFile1);
		_q_net2->load_from(QModelFile2);
		_policy_net->load_from(PModelFile);
	}

	// Copy over params
	std::stringstream stream1;
	_q_net1->save_to(stream1);
	_target_q_net1->load_from(stream1);

	std::stringstream stream2;
	_q_net2->save_to(stream2);
	_target_q_net2->load_from(stream2);

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

torch::Tensor SACAgent::get_action(torch::Tensor state)
{

	at::TensorList results = _policy_net->forward(state);
	at::Tensor mean = results[0];
	at::Tensor log_std = results[1];
	at::Tensor std = log_std.exp();

	Normal::Normal normal = Normal(mean, std);
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

//void getStateDict() {
//
//}

//void SaveStateDict(const torch::nn::Module& module, const std::string& file_name) {
//	torch::serialize::OutputArchive archive;
//	auto params = module.named_parameters(true /*recurse*/);
//	auto buffers = module.named_buffers(true /*recurse*/);
//	for (const auto& val : params) {
//		archive.write(val.key(), val.value());
//	}
//	for (const auto& val : buffers) {
//		archive.write(val.key(), val.value(), /*is_buffer*/ true);
//	}
//
//	archive.save_to(file_name);
//}
//
//void LoadStateDict(torch::nn::Module& module, const std::string& file_name) {
//	torch::serialize::InputArchive archive;
//	archive.load_from(file_name);
//	torch::NoGradGuard no_grad;
//	std::smatch m;
//	auto params = module.named_parameters(true);
//	auto buffers = module.named_buffers(true);
//	for (auto& val : params) {
//		archive.read(val.key(), val.value());
//	}
//	for (auto& val : buffers) {
//		archive.read(val.key(), val.value(), true);
//	}
//}

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
	_transfer_params_v2(*_q_net1, *_target_q_net1);
	_transfer_params_v2(*_q_net2, *_target_q_net2);

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

void SACAgent::_transfer_params_v1(torch::nn::Module& from, torch::nn::Module& to) {
	// char readBuffer[65536];
	std::stringstream stream("...");
	_save_to(from, stream);
	_load_from(to, stream);
}

void SACAgent::_save_to(torch::nn::Module& module, std::stringstream& fd) {
	
	torch::autograd::GradMode::set_enabled(false);
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
	torch::autograd::GradMode::set_enabled(true);
}

void SACAgent::_transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing) {
	torch::autograd::GradMode::set_enabled(false);

	auto to_params = to.named_parameters(true);
	auto from_params = from.named_parameters(true);

	for (auto& from_param : from_params) {
		torch::Tensor new_value = from_param.value();

		if (param_smoothing) {
			torch::Tensor old_value = to_params[from_param.key()];
			new_value = _tau * new_value + (1 - _tau) * old_value;
		} 
		
		to_params[from_param.key()].copy_(new_value);
	}

	torch::autograd::GradMode::set_enabled(true);
}

void SACAgent::_load_from(torch::nn::Module& module, std::stringstream& fd) {
	torch::autograd::GradMode::set_enabled(false);
	torch::serialize::InputArchive archive;
	archive.load_from(fd);
	torch::AutoGradMode enable_grad(false);
	auto params = module.named_parameters(true);
	auto buffers = module.named_buffers(true);
	for (auto& val : params) {
		archive.read(val.key(), val.value());
	}
	for (auto& val : buffers) {
		archive.read(val.key(), val.value(), true);
	}
	torch::autograd::GradMode::set_enabled(true);
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

	at::Tensor results = _policy_net->forward(state);
	
	at::Tensor mean = results[0];
	at::Tensor log_std = results[1];
	at::Tensor std = torch::exp(log_std);

	Normal normal = Normal(mean, std);
	torch::Tensor z = normal.sample();
	z.set_requires_grad(false);
	torch::Tensor actions = torch::tanh(z);
	
	return actions;
}

void SACAgent::update(int batchSize, Buffer* replayBuffer)
{
	// Generate training sample
	std::random_shuffle(replayBuffer->begin(), replayBuffer->end());
	
	double states[batchSize][_num_inputs];
	double next_states[batchSize][_num_inputs];
	double actions[batchSize][_num_actions];
	double rewards[batchSize]; 
	double dones[batchSize];

	for (int entry = 0;  entry < batchSize; entry++) {
		TD train_data = replayBuffer->at(entry);

		for (int i = 0; i < _num_inputs; i++) {
			states[entry][i] = train_data.currentState.stateArray[i];
			next_states[entry][i] = train_data.nextState.stateArray[i];

			if (i < _num_actions) {
				actions[entry][i] = train_data.actions[i];
			}
		}

		rewards[entry] = train_data.reward;
		dones[entry] = static_cast<double>(train_data.done);
	}

	// Prepare Training tensors
	auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU, -1);
	at::Tensor states_t = torch::from_blob(states, { batchSize, _num_inputs }, optionsDouble);
	at::Tensor next_states_t = torch::from_blob(next_states, { batchSize, _num_inputs }, optionsDouble);
	at::Tensor actions_t = torch::from_blob(actions, { batchSize, _num_actions }, optionsDouble);
	at::Tensor rewards_t = torch::from_blob(rewards, { batchSize }, optionsDouble);
	at::Tensor dones_t = torch::from_blob(dones, { batchSize }, optionsDouble);

	at::Tensor next = _policy_net->sample(next_states_t, batchSize);
	at::Tensor reshapedResult = next.view({ 2, batchSize, _num_actions });
	at::Tensor next_actions_t = reshapedResult[0];
	at::Tensor next_log_pi_t = reshapedResult[1];
	next_log_pi_t = next_log_pi_t.sum(1, true);
	
	// Predicted rewards
	at::Tensor next_q1s = _target_q_net1->forward(next_states_t, next_actions_t);
	at::Tensor next_q2s = _target_q_net2->forward(next_states_t, next_actions_t);

	// Conservative estimate of the value of the next state
	at::Tensor next_q_target_t = torch::min(next_q1s, next_q2s) - _alpha * next_log_pi_t;
	at::Tensor estimated_future_rewards = torch::unsqueeze(1.0 - dones_t, 1) * _gamma * next_q_target_t;
	at::Tensor expected_qs = torch::unsqueeze(rewards_t, 1) + estimated_future_rewards;

	// Q-Value loss
	at::Tensor curr_q1s = _q_net1->forward(states_t, actions_t);
	at::Tensor curr_q2s = _q_net2->forward(states_t, actions_t);
	at::Tensor q1_loss = torch::nn::functional::mse_loss(curr_q1s, expected_qs.detach());
	at::Tensor q2_loss = torch::nn::functional::mse_loss(curr_q2s, expected_qs.detach());

	// update Q-Value networks
	_q_net1->optimizer->zero_grad();
	q1_loss.backward();
	_q_net1->optimizer->step();

	_q_net2->optimizer->zero_grad();
	q2_loss.backward();
	_q_net2->optimizer->step();

	// Target Q-Value and Policy Network updates (delayed)
	at::Tensor current = _policy_net->sample(states_t, batchSize);
	at::Tensor reshapedCurrent = current.view({ 2, batchSize, _num_actions });
	at::Tensor pred_actions_t = reshapedCurrent[0];
	at::Tensor pred_log_pi_t = reshapedCurrent[1];

	if (_current_update == _max_delay) {
		_current_update = 0;
		at::Tensor min_q = torch::min(
			_q_net1->forward(states_t, pred_actions_t),
			_q_net2->forward(states_t, pred_actions_t)
		);

		at::Tensor policy_loss = (_alpha * pred_log_pi_t - min_q).mean();
		
		_policy_net->optimizer->zero_grad();
		policy_loss.backward();
		_policy_net->optimizer->step();

		// Copy over network params with averaging
		_transfer_params_v2(*_q_net1, *_target_q_net1, true);
		_transfer_params_v2(*_q_net2, *_target_q_net2, true);

		// Update alpha temperature
		at::Tensor alpha_loss = (_log_alpha * (-pred_log_pi_t - _target_entropy).detach()).mean();

		_alpha_optimizer->zero_grad();
		alpha_loss.backward();
		_alpha_optimizer->step();
		_alpha = _log_alpha.exp().item().toDouble();

	}
	else {
		_current_update++;
	}
}

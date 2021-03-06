﻿#include "SACAgent.h"
#include <bits/stdc++.h> 
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "Normal.h"
#include "util.h"
#include "data.h"

SACAgent::SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max, double action_min, bool alphaAdjuster, double gamma, double tau, double alpha, double q_lr, double p_lr, double a_lr)
{
	_self_adjusting_alpha = alphaAdjuster;
	_num_inputs = num_inputs;
	_num_actions = num_actions;
	_action_max = action_max;
	_action_min = action_min;

	_gamma = gamma;
	_tau = tau;
	_alpha = torch::tensor(alpha);
	_a_lr = a_lr;
	_q_lr = q_lr;
	_p_lr = p_lr;

	// initialize networks
	_q_net1 = new QNetwork(num_inputs, num_actions, num_hidden);
	_q_net2 = new QNetwork(num_inputs, num_actions, num_hidden);
	_policy_net = new PolicyNetwork(num_inputs, num_actions, num_hidden);
	_value_network = new ValueNetwork(num_inputs, num_hidden);
	_target_value_network = new ValueNetwork(num_inputs, num_hidden);
	_target_entropy = -1 * num_actions;
	

	// Load last checkpoint if available
	if (load_checkpoint()) {

	}
	else {
		_log_alpha = torch::log(_alpha);
		_log_alpha.set_requires_grad(true);
	}

	if (_self_adjusting_alpha) {
		// Auto Entropy adjustment variables
		_alpha_optimizer = new torch::optim::Adam({ _log_alpha }, torch::optim::AdamOptions(a_lr));
	}
	
	// Copy over params
	_transfer_params_v2(*_value_network, *_target_value_network);

}

SACAgent::~SACAgent() {

	delete _q_net1;
	delete _q_net2;
	delete _policy_net;
	delete _alpha_optimizer;
	delete _value_network;
	delete _target_value_network;
}

int SACAgent::sync(bool parent, double* data)
{
	if (parent) {
		int counter = 0;

		counter = this->_load_from_array(*_q_net1, data, counter);
		counter = this->_load_from_array(*_q_net2, data, counter);
		counter = this->_load_from_array(*_policy_net, data, counter);
		counter = this->_load_from_array(*_value_network, data, counter);
		counter = this->_load_from_array(*_target_value_network, data, counter);
		counter = this->_load_from_array(_log_alpha, data, counter);

		return counter;
	}
	else {
		int counter = 0;

		counter = this->_save_to_array(*_q_net1, data, counter);
		counter = this->_save_to_array(*_q_net2, data, counter);
		counter = this->_save_to_array(*_policy_net, data, counter);
		counter = this->_save_to_array(*_value_network, data, counter);
		counter = this->_save_to_array(*_target_value_network, data, counter);
		counter = this->_save_to_array(_log_alpha, data, counter);

		return counter;
	}
}

int SACAgent::_save_to_array(torch::nn::Module& from, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);

	auto params = from.named_parameters(true);
	auto buffers = from.named_buffers(true);
	
	for (const auto& val : params) {
		
		torch::Tensor value = val.value().clone().detach();
		value = torch::flatten(value);

		int size = value.size(-1);
		for (int i = 0; i < size; i++) {
			address[index] = value[i].item().toDouble();
			index++;
		}
	}
	
	for (const auto& val : buffers) {

		torch::Tensor value = val.value().clone().detach();	
		value = torch::flatten(value);

		int size = value.size(-1);
		for (int i = 0; i < size; i++) {
			address[index] = value[i].item().toDouble();
			index++;
		}
	}

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_save_to_array(torch::Tensor& from, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);

	address[index] = from.data().item().toDouble();
	index++;

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_load_from_array(torch::nn::Module& to, double* address, int index) {
	auto optionsDouble = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU, -1);
	torch::autograd::GradMode::set_enabled(false);

	auto params = to.named_parameters(true);
	auto buffers = to.named_buffers(true);
	for (const auto& val : params) {

		int size = val.value().numel();

		double copy[size];

		for (int i = 0; i < size; i++) {
			copy[i] = address[index];
			index++;
		}

		params[val.key()].data().copy_(torch::from_blob(copy, val.value().data().sizes(), optionsDouble));

	}
	
	for (const auto& val : buffers) {

		int size = val.value().numel();

		double copy[size];

		for (int i = 0; i < size; i++) {
			copy[i] = address[index];
			index++;
		}

		buffers[val.key()].data().copy_(torch::from_blob(copy, val.value().data().sizes(), optionsDouble));

	}

	torch::autograd::GradMode::set_enabled(true);
	return index;
}

int SACAgent::_load_from_array(torch::Tensor& to, double* address, int index) {
	torch::autograd::GradMode::set_enabled(false);

	to.data().copy_(torch::tensor(address[index]));

	torch::autograd::GradMode::set_enabled(true);
	index++;
	return index;
}

void SACAgent::_transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing) {
	torch::autograd::GradMode::set_enabled(false);

	auto to_params = to.named_parameters(true);
	auto from_params = from.named_parameters(true);

	for (auto& from_param : from_params) {
		torch::Tensor new_value = from_param.value();

		if (param_smoothing) {
			torch::Tensor old_value = to_params[from_param.key()];
			new_value = _tau * new_value + (1.0 - _tau) * old_value;
		} 
		
		to_params[from_param.key()].data().copy_(new_value);
	}

	torch::autograd::GradMode::set_enabled(true);
}

void SACAgent::save_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = path + "/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = path + "/Q_Net_Checkpoint2.pt";
	std::string PModelFile = path + "/P_Net_Checkpoint.pt";
	std::string AlphaFile = path + "/Alpha_Checkpoint.pt";
	std::string ValueFile = path + "/Value_Checkpoint.pt";
	std::string TargetValueFile = path + "/Target_Value_Checkpoint.pt";

	torch::serialize::OutputArchive QModelArchive1;
	_q_net1->save(QModelArchive1);
	QModelArchive1.save_to(QModelFile1);

	torch::serialize::OutputArchive QModelArchive2;
	_q_net2->save(QModelArchive2);
	QModelArchive2.save_to(QModelFile2);

	torch::serialize::OutputArchive ValueArchive;
	_value_network->save(ValueArchive);
	ValueArchive.save_to(ValueFile);

	torch::serialize::OutputArchive TargetValueArchive;
	_value_network->save(TargetValueArchive);
	ValueArchive.save_to(TargetValueFile);

	torch::serialize::OutputArchive PModelArchive;
	_policy_net->save(PModelArchive);
	PModelArchive.save_to(PModelFile);

	torch::save(_log_alpha, AlphaFile);
}

bool SACAgent::load_checkpoint()
{
	// Load from file if exists
	std::string path = get_current_dir_name();
	std::string QModelFile1 = path + "/Q_Net_Checkpoint1.pt";
	std::string QModelFile2 = path + "/Q_Net_Checkpoint2.pt";
	std::string PModelFile = path + "/P_Net_Checkpoint.pt";
	std::string AlphaFile = path + "/Alpha_Checkpoint.pt";
	std::string ValueFile = path + "/Value_Checkpoint.pt";
	std::string TargetValueFile = path + "/Target_Value_Checkpoint.pt";


	if (
			Utility::fileExists(QModelFile1) && 
			Utility::fileExists(QModelFile2) && 
			Utility::fileExists(PModelFile) &&
			Utility::fileExists(AlphaFile) &&
			Utility::fileExists(ValueFile) &&
			Utility::fileExists(TargetValueFile)
		) 
	{
		torch::serialize::InputArchive QModelArchive1;
		QModelArchive1.load_from(QModelFile1);
		_q_net1->load(QModelArchive1);

		torch::serialize::InputArchive QModelArchive2;
		QModelArchive2.load_from(QModelFile2);
		_q_net2->load(QModelArchive2);

		torch::serialize::InputArchive PModelArchive;
		PModelArchive.load_from(PModelFile);
		_policy_net->load(PModelArchive);

		torch::serialize::InputArchive ValueArchive;
		ValueArchive.load_from(ValueFile);
		_value_network->load(ValueArchive);

		torch::serialize::InputArchive TargetValueArchive;
		TargetValueArchive.load_from(TargetValueFile);
		_target_value_network->load(TargetValueArchive);

		torch::load(_log_alpha, AlphaFile);
		return true;
	}
	else {
		return false;
	}
}

torch::Tensor SACAgent::get_action(torch::Tensor state)
{
	at::Tensor results;

	if (pthread_mutex_lock(&_policyNetLock) == 0) {
		results = _policy_net->forward(state);
		pthread_mutex_unlock(&_policyNetLock);
	}
	else {
		throw "Could not obtain lock";
	}
	
	at::Tensor mean = results[0];
	at::Tensor log_std = results[1];
	at::Tensor std = torch::exp(log_std);

	Normal normal = Normal(mean, std);
	torch::Tensor z = normal.sample();
	z.set_requires_grad(false);
	torch::Tensor actions = torch::tanh(z);
	
	return actions;
}

void SACAgent::update(int batchSize, TrainBuffer* replayBuffer)
{
	
	double states[batchSize][_num_inputs];
	double next_states[batchSize][_num_inputs];
	double actions[batchSize][_num_actions];
	double rewards[batchSize]; 
	double dones[batchSize];
	double currentStateArray[_num_inputs];
	double nextStateArray[_num_inputs];

	for (int entry = 0;  entry < batchSize; entry++) {
		TD train_data = replayBuffer->at(entry);
		train_data.currentState.getStateArray(currentStateArray);
		train_data.nextState.getStateArray(nextStateArray);

		for (int i = 0; i < _num_inputs; i++) {
			states[entry][i] = currentStateArray[i];
			next_states[entry][i] = nextStateArray[i];

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

	// Sample from Policy
	at::Tensor next = _policy_net->sample(states_t, batchSize);
	at::Tensor reshapedResult = next.view({ 5, batchSize, _num_actions });
	at::Tensor next_actions_t = reshapedResult[0];
	at::Tensor next_log_pi_t = reshapedResult[1];
	at::Tensor next_mean = reshapedResult[2];
	at::Tensor next_std = reshapedResult[3];
	at::Tensor next_z_values = reshapedResult[4];
	next_log_pi_t = next_log_pi_t.sum(1, true);

	// Update alpha temperature
	if (_self_adjusting_alpha) {

		at::Tensor alpha_loss = (-_log_alpha * (next_log_pi_t + _target_entropy).detach()).mean();
		_alpha_optimizer->zero_grad();
		alpha_loss.backward();
		_alpha_optimizer->step();
		_alpha = torch::exp(_log_alpha);
		// std::cout << "Current alpha: " << _alpha << std::endl;
	}
	
	// Estimated Q-Values
	at::Tensor predicted_q_value1 = _q_net1->forward(states_t, actions_t);
	at::Tensor predicted_q_value2 = _q_net2->forward(states_t, actions_t);
	at::Tensor predicted_values = _value_network->forward(states_t);

	// Training the Q-Value Function
	at::Tensor target_values = _target_value_network->forward(next_states_t);
	at::Tensor target_q_values = torch::unsqueeze(rewards_t, 1) + torch::unsqueeze(1.0 - dones_t, 1) * _gamma * target_values;
	
	at::Tensor q_value_loss1 = torch::nn::functional::mse_loss(predicted_q_value1, target_q_values.detach());
	at::Tensor q_value_loss2 = torch::nn::functional::mse_loss(predicted_q_value2, target_q_values.detach());

	// Training Value Function
	at::Tensor predicted_new_q_value = torch::min(predicted_q_value1, predicted_q_value2);
	at::Tensor target_value_func = predicted_new_q_value - _alpha * next_log_pi_t;
	at::Tensor value_loss = torch::nn::functional::mse_loss(predicted_values, target_value_func.detach());
	
	/*at::Tensor policy_loss = (next_log_pi_t - predicted_new_q_value).mean();
	
	if (pthread_mutex_lock(&_policyNetLock) == 0) {
	_policy_net->zero_grad();
	policy_loss.backward();
	_policy_net->optimizer->step();
	_transfer_params_v2(*_value_network, *_target_value_network, true);
	*/

	// Update Q-Value networks
	_q_net1->optimizer->zero_grad();
	q_value_loss1.backward();
	_q_net1->optimizer->step();

	_q_net2->optimizer->zero_grad();
	q_value_loss2.backward();
	_q_net2->optimizer->step();

	// Update Value network
	_value_network->zero_grad();
	value_loss.backward();
	_value_network->optimizer->step();

	// Delay update of Target Value and Policy Networks
	if (_current_update == _max_delay) {
		_current_update = 0;

		// Train Policy Network 
		at::Tensor advantages = torch::min(_q_net1->forward(states_t, actions_t), _q_net2->forward(states_t, actions_t)) - predicted_values.detach();
		at::Tensor policy_loss = (_alpha * next_log_pi_t.sum(1, true) - advantages).mean();

		// Regularization
		at::Tensor mean_reg = 1e-3 * next_mean.pow(2).sum(1, true).mean();
		at::Tensor std_reg = 1e-3 * next_std.pow(2).sum(1, true).mean();
		// at::Tensor z_value_reg = 1e-3 * next_z_values.pow(2).sum(-1).mean();

		at::Tensor actor_reg = mean_reg + std_reg;
		policy_loss += actor_reg;

		std::cout << "Policy Loss: " << policy_loss << std::endl;

		// Update Policy Network
		if (pthread_mutex_lock(&_policyNetLock) == 0) {
			_policy_net->optimizer->zero_grad();
			policy_loss.backward();
			_policy_net->optimizer->step();
			pthread_mutex_unlock(&_policyNetLock);
		}
		else {
			throw "could not obtain lock";
		}

		// Copy over network params with averaging
		_transfer_params_v2(*_value_network, *_target_value_network, true);

		if (_current_save_delay == _max_save_delay) {
			_current_save_delay = 0;
			save_checkpoint();
		}
		
	}
	else {
		_current_save_delay++;
		_current_update++;
	}

}


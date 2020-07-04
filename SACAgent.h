#pragma 
#include <torch/torch.h>
#include "data.h"
#include "PolicyNetwork.h"
#include "QNetwork.h"

class SACAgent
{
private:
	double _gamma, _tau, _alpha, _a_lr, _q_lr, _p_lr;
	int _num_inputs, _num_actions;
	double _action_max, _action_min;

	QNetwork::QNetwork* _q_net1; 
	QNetwork::QNetwork* _q_net2;
	QNetwork::QNetwork* _target_q_net1;
	QNetwork::QNetwork* _target_q_net2;
	PolicyNetwork::PolicyNetwork* _policy_net;

	torch::Tensor _log_alpha;
	c10::Scalar _target_entropy;
	torch::optim::Adam* _alpha_optimizer = nullptr;

public:
	SACAgent(int num_inputs, int num_actions, double action_max, double action_min, double gamma = 0.99, double tau = 0.01, double alpha = 0.2, double q_lr = 3e-4, double policy_lr = 3e-4, double a_lr = 3e-4);
	~SACAgent();
	void update(int batchSize, Buffer* replayBuffer);
	torch::Tensor get_action(torch::Tensor state);
};


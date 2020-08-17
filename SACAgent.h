#pragma 
#include <torch/torch.h>
#include "data.h"
#include "PolicyNetwork.h"
#include "QNetwork.h"
#include "ValueNetwork.h"

class SACAgent
{
private:
	double _gamma, _tau, _a_lr, _q_lr, _p_lr;
	int _num_inputs, _num_actions;
	double _action_max, _action_min;
	int _current_update = 0;
	int _current_save_delay = 0;
	int _max_save_delay = 10;
	int _max_delay = 2;

	bool _self_adjusting_alpha;
	
	// For internal syncing of access
	pthread_mutex_t _policyNetLock = PTHREAD_MUTEX_INITIALIZER;

	QNetwork* _q_net1; 
	QNetwork* _q_net2;
	ValueNetwork* _value_network;
	ValueNetwork* _target_value_network;
	PolicyNetwork* _policy_net;

	torch::Tensor _log_alpha, _alpha;
	c10::Scalar _target_entropy;
	torch::optim::Adam* _alpha_optimizer = nullptr;
	void _save_to(torch::nn::Module& model, std::stringstream& fd);
	void _load_from(torch::nn::Module& model, std::stringstream& fd);
	void _transfer_params_v1(torch::nn::Module& from, torch::nn::Module& to);
	void _transfer_params_v2(torch::nn::Module& from, torch::nn::Module& to, bool param_smoothing = false);

public:
	SACAgent(int num_inputs, int num_hidden, int num_actions, double action_max = 1.0, double action_min = 0.0, bool alphaAdjuster = true, double gamma = 0.99, double tau = 0.01, double alpha = 0.2, double q_lr = 3e-4, double policy_lr = 3e-4, double a_lr = 3e-4);
	~SACAgent();

	void update(int batchSize, TrainBuffer* replayBuffer);
	torch::Tensor get_action(torch::Tensor state);

	void save_checkpoint();
	bool load_checkpoint();

};


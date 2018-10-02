import torch
from torch.optim import Adam

from robo_rl.sac.tanh_gaussian_policy import TanhGaussianPolicy
from robo_rl.common.networks.value_network import QNetwork, ValueNetwork

# ASSUMPTION : non deterministic
def soft_update(original , target, t):
    # zip(a,b) is same as [a.b]
    for original_param,target_param  in zip(original.parameters(), target.parameters()):
        target_param.data.copy_(original_param.data*t + target_param*(1-t))
        ## check copy_ parameter : Something on cpu or gpu, also no need for return as it changes in self

def hard_update(original,target):
    for original_param,target_param in zip(original.parameters(),target.parameters()):
        target_param.data.copy_(original_param.data)


class SAC:
    def __init__(self,action_dim,state_dim,**args ):   ## TO do
        self.action_dim= action_dim
        self.state_dim=state_dim
        self.gamma= args['gamma']
        self.scale_reward=args['scale_reward']
        self.reparam = args['reparam']
        self.deterministic= args['deterministic']
        self.target_update_interval = args['target_update_interval']
        self.hidden_dim = args['hidden_dim']
        self.lr=args['lr']

        ## hard and soft updates



        self.value =ValueNetwork(self.state_dim,self.hidden_dim)
        self.value_target =ValueNetwork(self.state_dim,self.hidden_dim)
        self.value_optimizer=Adam(self.value.parameters(),lr=self.lr)
        hard_update(self.value,self.value_target)


        self.policy = TanhGaussianPolicy(self.state_dim,self.action_dim, self.hidden_dim)
        self.policy_optimizer= Adam(self.policy.parameters(), lr=self.lr)

        self.critic = QNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_optimizer- Adam(self.policy.parameters(),lr=self.lr)


## batch is dict from replay buffer
    def policy_update(self,batch):
        state_batch=torch.FloatTensor(batch['state'])
        action_batch=torch.FloatTensor(batch['action'])
        next_state_batch=batch['next_state']
        done_batch=batch['done']

        q1, q2= self.critic(state_batch,action_batch)

        # now to stabilize training for soft value
        # actions sampled according to current policy and not replay buffer
        # use minimum of 2 q values for value grad and policy grad

        # assuming stochaistic
        value = self.value(state_batch)
        target_value= self.value_target(next_state_batch)








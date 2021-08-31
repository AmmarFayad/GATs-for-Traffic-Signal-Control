import numpy as np
import torch
from marl.rlcore.algo import JointPPO
from marl.rlagent import Neo
from marl.mpnn import MPNN

def setup_master(args, return_env=False):
    policy = None
    team = []
    # share a common policy in a team
    #print("ACTIONS", args.num_actions)
    action_space = args.num_actions
    entity_mp = args.entity_mp
    if args.env_name == 'graph-atl':
        num_entities = 0
    else:
        raise NotImplementedError('Unknown environment, define entity_mp for this!')

    if entity_mp:
        pol_obs_dim = args.obs_size - 2 * num_entities
    else:
        pol_obs_dim = args.obs_size

    # index at which agent's position is present in its observation
    pos_index = pol_obs_dim
    for i in range(args.num_agents):
        obs_dim = args.obs_size
        args.mask_dist=1
        if policy is None:
                policy = MPNN(input_size=pol_obs_dim,num_agents=args.num_agents,num_entities=num_entities,action_space=action_space,
                               pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp, num_actions=args.num_actions).to(args.device)
        team.append(Neo(args,policy,(obs_dim,),action_space))
    master = Learner(args, [team], [policy])
    
    if args.continue_training:
        print("Loading pretrained model")
        master.load_models(torch.load(args.load_dir)['models'])

    return master


class Learner(object):
    # supports centralized training of agents in a team
    def __init__(self, args, teams_list, policies_list):
        self.teams_list = [x for x in teams_list if len(x)!=0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]
        self.trainers_list = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]
        self.device = args.device

    @property
    def all_policies(self):
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        return self.policies_list[0].attn_mat

    def initialize_obs(self, obs):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[:,i,:]).float().to(self.device))
            agent.rollouts.to(self.device)

    def act(self, step):
        actions_list = []
        for team, policy in zip(self.teams_list, self.policies_list):
            # concatenate all inputs
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team])
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team])
            all_masks = torch.cat([agent.rollouts.masks[step] for agent in team])

            props = policy.act(all_obs, all_hidden, all_masks, deterministic=False) # a single forward pass 

            # split all outputs
            n = len(team)
            all_value, all_action, all_action_log_prob, all_states = [torch.chunk(x, n) for x in props]
            for i in range(n):
                team[i].value = all_value[i]
                team[i].action = all_action[i]
                team[i].action_log_prob = all_action_log_prob[i]
                team[i].states = all_states[i]
                actions_list.append(all_action[i].cpu().numpy())

        return actions_list

    def update(self):
        return_vals = []
        # use joint ppo for training each team
        for i, trainer in enumerate(self.trainers_list):
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            vals = trainer.update(rollouts_list)
            return_vals.append([np.array(vals)]*len(rollouts_list))
        
        return np.stack([x for v in return_vals for x in v]).reshape(-1,3)

    def wrap_horizon(self):
        for team, policy in zip(self.teams_list,self.policies_list):
            last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
            last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[-1] for agent in team])
            last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])
            
            with torch.no_grad():
                next_value = policy.get_value(last_obs, last_hidden, last_masks)

            all_value = torch.chunk(next_value,len(team))
            for i in range(len(team)):
                team[i].wrap_horizon(all_value[i])

    def after_update(self):
        for agent in self.all_agents:
            agent.after_update()

    def update_rollout(self, obs, reward, masks):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t[:, i, :]
            agent.update_rollout(agent_obs, reward[:,i].unsqueeze(1), masks[:,i].unsqueeze(1))

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def set_eval_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.train()

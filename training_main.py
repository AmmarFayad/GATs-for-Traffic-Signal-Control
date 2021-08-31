from __future__ import absolute_import
from __future__ import print_function

import datetime

from wandb import env
#from training_simulation import Simulation

from utils import  set_sumo
import wandb

# marl related imports
from marl.arguments import get_args
from marl.learner import setup_master
import random
import torch
import numpy as np
from pprint import pprint
import json
import os
from copy import deepcopy

if __name__ == "__main__":
   
    episode = 0
    timestamp_start = datetime.datetime.now()

    

    # marl related configurations
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    master = setup_master(args) 
    print(str(args.num_agents)+'intersection/network.sumo.cfg')
    args.sumocfg_file_name=str(args.num_agents)+'intersection/network.sumo.cfg'
    sumo_cmd = set_sumo(args.gui, args.sumocfg_file_name, args.max_steps)

    project = "Graph ATL %s Agents" %(args.num_agents)
    wandb.init(project=project)

    newpath = r'saved/run_%s_AGENTS=%s' %(datetime.datetime.now().strftime("%Y-%m-%d_%H"), args.num_agents) 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    if args.num_agents==7:
        from generator3 import TrafficGenerator
    elif args.num_agents==3:
        from generator2 import TrafficGenerator
    elif args.num_agents==1:
        from generator1 import TrafficGenerator

    TrafficGen = TrafficGenerator(
        args.max_steps, 
        args.n_cars_generated
    )
    
    if args.algo_paradigm=='Interaction-Graph': from training_simulation import Simulation
    elif args.algo_paradigm=='fixed': from baselines.fixed import Simulation
    elif args.algo_paradigm=='pressure': from baselines.maxpressure import Simulation
    Simulation = Simulation(
        TrafficGen,
        sumo_cmd,
        args.max_steps, 
        args.green_duration,
        args.yellow_duration,
        args.obs_size,
        args.num_actions,
        master
    )

    while episode < args.total_episodes:
        print('\n [INFO]----- Episode', str(episode + 1), '/', str(args.total_episodes), '-----')
        # run the simulation
        simulation_time, training_time, avg_reward, avg_waiting, training_loss, dist_entropy = Simulation.run(episode, args)
        print('\t [STAT] Simulation time:', simulation_time, 's - Training time:',
              training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
        # log the training progress in wandb
        wandb.log({
            "all/training_loss": np.sum(training_loss),
            "all/avg_reward": avg_reward,
            "all/avg_waiting_time": avg_waiting,
            "all/travel_time": simulation_time,
            "all/training_time": training_time,
            "all/entropy": np.sum(dist_entropy)}, step=episode)

        if episode % args.save_interval == 0:# and not args.test_run:
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None)
            savedict['ob_rms'] = ob_rms
            savedir = newpath +'/ep' +str(episode) + '.pt'
            torch.save(savedict, savedir)

        episode += 1


    print("\n [INFO] End of Training")
    print("\t [STAT] Start time:", timestamp_start)
    print("\t [STAT] End time:", datetime.datetime.now())
    print("\t [STAT] Session info saved at:", newpath)

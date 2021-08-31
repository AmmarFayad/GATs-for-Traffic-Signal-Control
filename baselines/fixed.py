#from math import dist
import traci
import numpy as np
import random
import timeit
import torch
import torch.optim as optim
from torch.autograd import Variable
from utils import set_sumo

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration,
                 yellow_duration, num_states, num_actions, master):
        self.sum_waiting_time = 0
        self.sum_queue_length = 0
        self.waiting_times = {}
        self.traffic_gen = TrafficGen
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self.master = master
    def run(self, episode, args):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # generate the route file for this simulation and set up sumo
        self.traffic_gen.generate_routefile(seed=episode)
        #if episode>130: self.sumo_cmd=set_sumo(True, args.sumocfg_file_name, args.max_steps)
        traci.start(self.sumo_cmd)

        print("\t [INFO] Start simulating the episode")

        # inits
        if args.num_agents==3: junctions=["B1", "C1", "D1"]
        elif args.num_agents==1: junctions=["B1"]
        elif args.num_agents==7: junctions=["B1", "C1", "D1", "E1", "F1", "G1", "H1"]
        self.step = 0
        old_total_wait = 0
        old_action = [-1 for i in range (args.num_agents)]

        sum_reward = 0
        sum_waiting = 0
        
        while self.step < args.num_steps:

            # get current state of the intersection
            current_state = self.get_state_general(args,junctions)
            current_state = current_state.reshape(1, args.num_agents, args.obs_size)
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawned in the environment,
            current_total_wait = self.collect_waiting_times(args.num_agents)
            reward = old_total_wait - current_total_wait
            reward*=1e-2 #Reshape the rewards
            reward = (torch.as_tensor(np.array([[reward] for i in range (args.num_agents)]))).view(1, -1)
            
            # # initialize the master rollout
            # if self.step == 0:
            #     self.master.initialize_obs(current_state)
            # else:
            #     # update the master rollout
            #     masks = torch.FloatTensor([[1] for i in range (args.num_agents)]).to(args.device)
            #     masks = masks.view(1, -1)
            #     self.master.update_rollout(current_state, reward, masks)
                
            
            # choose the light phase to activate, based on the current state of the intersection
            with torch.no_grad():
                if self.step==0:
                    l=random.randint(0,1)
                    action = [np.array([l]) for i in range(args.num_agents)] #self.master.act(self.step)
                else: 
                    action=[np.array([1-action[i][0]]) for i in range(args.num_agents)]
            #print(action, action[0])
                
            
            for i, junc in enumerate(junctions):
                # if the chosen phase is different from the last phase, activate the yellow phase
                if self.step != 0 and old_action[i] != action[i][0]:
                    self.set_yellow_phase(id=junc,old_action=action[i][0])
                    #self.simulate(junc, self.yellow_duration, is_green=False)
            self.simulate(junctions, self.yellow_duration, is_green=False)    
            for i, junc in enumerate(junctions):
                # execute the phase selected before
                self.set_green_phase(id=junc,action_number=action[i][0])
                #self.simulate(junc,self.green_duration, is_green=True)
                
                # saving variables for later & accumulate reward
                old_action[i] = action[i][0]
            self.simulate(junctions, self.green_duration, is_green=False) 
            old_total_wait = current_total_wait
            self.step += 1
            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward[0][0] < 0:
                sum_reward += reward[0][0]
            sum_waiting += current_total_wait
            

        avg_reward = sum_reward / self.max_steps
        avg_waiting = sum_waiting / self.max_steps
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("\t [STAT] Average reward:", avg_reward, "Average waiting time:", avg_waiting)

        print("\t [INFO] Training the model")
        start_time = timeit.default_timer()
        # self.master.wrap_horizon()
        # return_vals = self.master.update()
        # value_loss = return_vals[:, 0]
        # action_loss = return_vals[:, 1]
        # dist_entropy = return_vals[:, 2]
        # self.master.after_update()

        # print("\t [STAT] Training value loss :", value_loss)
        # print("\t [STAT] Training action loss :", action_loss)
        # print("\t [STAT] Training entropy :", dist_entropy)
        training_time = round(timeit.default_timer() - start_time, 1)
        #total_loss = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef

        return simulation_time, training_time, avg_reward, avg_waiting,1,1# total_loss, dist_entropy

    def simulate(self, juncs, steps_todo, is_green):
        """
        Execute steps in sumo while gathering statistics
        """
        # do not do more steps than the maximum allowed number of steps
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step
        #if is_green: self.step += 1
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            #self.step += 1  # update the step counter
            steps_todo -= 1
            for id in juncs: queue_length = self.get_queue_length(id)
            self.sum_queue_length += queue_length
            # 1 step while waiting in queue means 1 second waited
            # for each car, therefore queue_length == waited_seconds
            self.sum_waiting_time += queue_length

    def collect_waiting_times(self, is_one):
        """
        Retrieve the waiting time of every car in the incoming roads and return the total waiting time
        """
        if is_one==1: incoming_roads=["A1B1", "B2B1", "C1B1", "B0B1"]#incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        elif is_one==3: incoming_roads=["A1B1", "B2B1", "C1B1", "B0B1", "B1C1", "C2C1", "D1C1", "C0C1", "C1D1", "D2D1", "D0D1", "E1D1"]
        elif is_one==7:
            incoming_roads=["A1B1", "B2B1", "C1B1", "B0B1", "B1C1", "C2C1", "D1C1", "C0C1", "C1D1", "D2D1", "D0D1", "E1D1"
            "D1E1","E2E1", "F1E1", "E0E1", "E1F1", "F2F1", "G1F1", "F0F1", "F1G1", "G2G1", "H1G1", "G0G1",
            "G1H1", "H2H1", "I1H1", "H0H1"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)
            # consider only the waiting times of cars in incoming roads
            if road_id in incoming_roads:
                self.waiting_times[car_id] = wait_time
            else:
                # a car that was tracked has cleared the intersection
                if car_id in self.waiting_times:
                    del self.waiting_times[car_id]
        total_waiting_time = sum(self.waiting_times.values())
        return total_waiting_time

    def set_yellow_phase(self, id, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        yellow_phase_code = old_action * 2 +1
        traci.trafficlight.setPhase(id, yellow_phase_code) #yellow_phase_code)

    def set_green_phase(self, id, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase(id, 0)#PHASE_NS_GREEN)      North SOuth
        elif action_number == 1:
            traci.trafficlight.setPhase(id, 2)#PHASE_EW_GREEN)      EW
        

    def get_queue_length(self,id):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        
        if id=="TL":
            halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
            halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
            halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
            halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="B1":
            halt_N = traci.edge.getLastStepHaltingNumber("A1B1")
            halt_S = traci.edge.getLastStepHaltingNumber("B2B1")
            halt_E = traci.edge.getLastStepHaltingNumber("C1B1")
            halt_W = traci.edge.getLastStepHaltingNumber("B0B1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="C1":
            halt_N = traci.edge.getLastStepHaltingNumber("B1C1")
            halt_S = traci.edge.getLastStepHaltingNumber("C2C1")
            halt_E = traci.edge.getLastStepHaltingNumber("D1C1")
            halt_W = traci.edge.getLastStepHaltingNumber("C0C1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="D1":
            halt_N = traci.edge.getLastStepHaltingNumber("C1D1")
            halt_S = traci.edge.getLastStepHaltingNumber("D2D1")
            halt_E = traci.edge.getLastStepHaltingNumber("E1D1")
            halt_W = traci.edge.getLastStepHaltingNumber("D0D1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="E1":
            halt_N = traci.edge.getLastStepHaltingNumber("D1E1")
            halt_S = traci.edge.getLastStepHaltingNumber("E2E1")
            halt_E = traci.edge.getLastStepHaltingNumber("F1E1")
            halt_W = traci.edge.getLastStepHaltingNumber("E0E1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="F1":
            halt_N = traci.edge.getLastStepHaltingNumber("E1F1")
            halt_S = traci.edge.getLastStepHaltingNumber("F2F1")
            halt_E = traci.edge.getLastStepHaltingNumber("G1F1")
            halt_W = traci.edge.getLastStepHaltingNumber("F0F1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="G1":
            halt_N = traci.edge.getLastStepHaltingNumber("F1G1")
            halt_S = traci.edge.getLastStepHaltingNumber("G2G1")
            halt_E = traci.edge.getLastStepHaltingNumber("H1G1")
            halt_W = traci.edge.getLastStepHaltingNumber("G0G1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
        if id=="H1":
            halt_N = traci.edge.getLastStepHaltingNumber("G1H1")
            halt_S = traci.edge.getLastStepHaltingNumber("H2H1")
            halt_E = traci.edge.getLastStepHaltingNumber("I1H1")
            halt_W = traci.edge.getLastStepHaltingNumber("H0H1")
            queue_length = halt_N + halt_S + halt_E + halt_W
            return queue_length
    
  
    def get_state_general(self,args,junctions):
        """
        Retrieve the state of multiple intersection from sumo, in the form of cell occupancy
        """
        state = np.array([np.zeros(self.num_states) for i in range(args.num_agents)])
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            
            if traci.vehicle.getNextTLS(car_id) is not () and traci.vehicle.getNextTLS(car_id)[0][0] in junctions:
                c=traci.vehicle.getNextTLS(car_id)
                lane_pos=c[0][2]
                
                
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 50:
                    lane_cell = 5
                elif lane_pos < 60:
                    lane_cell = 6
                elif lane_pos < 70:
                    lane_cell = 7
                elif lane_pos < 80:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell =  9
                
                if c[0][0]=="B1":
                    
                    if lane_id == "A1B1_0" :
                        lane_group = 0
                    elif lane_id == "B2B1_0":
                        lane_group = 1
                    elif lane_id == "C1B1_0":
                        lane_group = 2
                    elif lane_id == "B0B1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        # composition of the two position ID to create a number in interval 0-79
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        # flag for not detecting cars crossing the intersection or driving away from it
                        valid_car = False

                    if valid_car:
                        # write the position of the car car_id in the state array in the form of "cell occupied"
                        state[0][car_position] = 1
                elif c[0][0]=="C1":
                    if lane_id == "B1C1_0" :
                        lane_group = 0
                    elif lane_id == "C2C1_0":
                        lane_group = 1
                    elif lane_id == "D1C1_0":
                        lane_group = 2
                    elif lane_id == "C0C1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[1][car_position] = 1
                elif c[0][0]=="D1":
                    if lane_id == "C1D1_0" :
                        lane_group = 0
                    elif lane_id == "D2D1_0":
                        lane_group = 1
                    elif lane_id == "E1D1_0":
                        lane_group = 2
                    elif lane_id == "D0D1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[2][car_position] = 1
                
                elif c[0][0]=="E1":
                    if lane_id == "D1E1_0" :
                        lane_group = 0
                    elif lane_id == "E2E1_0":
                        lane_group = 1
                    elif lane_id == "F1E1_0":
                        lane_group = 2
                    elif lane_id == "E0E1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[3][car_position] = 1
                
                elif c[0][0]=="E1":
                    if lane_id == "E1F1_0" :
                        lane_group = 0
                    elif lane_id == "F2F1_0":
                        lane_group = 1
                    elif lane_id == "G1F1_0":
                        lane_group = 2
                    elif lane_id == "F0F1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[4][car_position] = 1
                
                elif c[0][0]=="G1":
                    if lane_id == "F1G1_0" :
                        lane_group = 0
                    elif lane_id == "G2G1_0":
                        lane_group = 1
                    elif lane_id == "H1G1_0":
                        lane_group = 2
                    elif lane_id == "G0G1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[5][car_position] = 1
                elif c[0][0]=="H1":
                    if lane_id == "G1H1_0" :
                        lane_group = 0
                    elif lane_id == "H2H1_0":
                        lane_group = 1
                    elif lane_id == "I1H1_0":
                        lane_group = 2
                    elif lane_id == "H0H1_0":
                        lane_group = 3
                    else:
                        lane_group = -1

                    if 1 <= lane_group <= 3:
                        car_position = int(str(lane_group) + str(lane_cell))
                        valid_car = True
                    elif lane_group == 0:
                        car_position = lane_cell
                        valid_car = True
                    else:
                        valid_car = False

                    if valid_car:
                        state[6][car_position] = 1
        return state





###############################################################################
#This is not used--To be deleted
    def get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self.num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            # invert lane position value,
            # so if the car is close to the traffic light -> lane_pos = 0 -> 750 = max len of a road
            # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getLanePosition
            #lane_pos = 750 - lane_pos

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 50:
                lane_cell = 5
            elif lane_pos < 60:
                lane_cell = 6
            elif lane_pos < 70:
                lane_cell = 7
            elif lane_pos < 80:
                lane_cell = 8
            elif lane_pos <= 90:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if 1 <= lane_group <= 7:
                # composition of the two position ID to create a number in interval 0-79
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                # flag for not detecting cars crossing the intersection or driving away from it
                valid_car = False

            if valid_car:
                # write the position of the car car_id in the state array in the form of "cell occupied"
                state[car_position] = 1

        return state



    def state_trial (self, args):
        state = np.array([np.zeros(self.num_states) for i in range(args.num_agents)])
        car_list = traci.vehicle.getIDList()
        
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            
            # invert lane position value,
            # so if the car is close to the traffic light -> lane_pos = 0 -> 750 = max len of a road
            # https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getLanePosition
            #lane_pos = 750 - lane_pos
            
            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 50:
                lane_cell = 5
            elif lane_pos < 60:
                lane_cell = 6
            elif lane_pos < 70:
                lane_cell = 7
            elif lane_pos < 80:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell =  9
            #print(lane_id, lane_cell)
            if lane_id[2]=="B":
                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "A1B1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "B2B1_0":
                    lane_group = 1
                elif lane_id == "C1B1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "B0B1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[0][car_position] = 1
            elif lane_id[2]=="C":
                ####################################################2nd intersection
                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "B1C1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "C2C1_0":
                    lane_group = 1
                elif lane_id == "D1C1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "C0C1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[1][car_position] = 1
            elif lane_id[2]=="D":
                ##################################################################3rd int

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "C1D1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "D2D1_0":
                    lane_group = 1
                elif lane_id == "E1D1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "D0D1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[2][car_position] = 1
            #####################################
            elif lane_id[2]=="E":
                ##################################################################3rd int

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "D1E1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "E2E1_0":
                    lane_group = 1
                elif lane_id == "F1E1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "E0E1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[3][car_position] = 1
            #####################################
            elif lane_id[2]=="F":
                ##################################################################3rd int

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "E1F1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "F2F1_0":
                    lane_group = 1
                elif lane_id == "G1F1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "F0F1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[4][car_position] = 1
            #######################################
            elif lane_id[2]=="G":
                ##################################################################3rd int

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "F1G1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "G2G1_0":
                    lane_group = 1
                elif lane_id == "H1G1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "G0G1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[5][car_position] = 1
            elif lane_id[2]=="H":
                ##################################################################3rd int

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "G1H1_0" :#or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "H2H1_0":
                    lane_group = 1
                elif lane_id == "I1H1_0":# or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "H0H1_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if 1 <= lane_group <= 3:
                    # composition of the two position ID to create a number in interval 0-79
                    car_position = int(str(lane_group) + str(lane_cell))
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    # flag for not detecting cars crossing the intersection or driving away from it
                    valid_car = False

                if valid_car:
                    # write the position of the car car_id in the state array in the form of "cell occupied"
                    state[6][car_position] = 1
        return state
###############################################################################
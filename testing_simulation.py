import traci
import numpy as np
import timeit
import torch
from torch.autograd import Variable

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
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self.dqn = Model
        self.traffic_gen = TrafficGen
        self.step = 0
        self.sumo_cmd = sumo_cmd
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions

    def run(self, episode):
        # for the purpose of recording trajectories
        f = open("data.txt", "a")

        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self.traffic_gen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)
        print("Simulating...")

        # inits
        self.step = 0
        self.waiting_times = {}
        old_total_wait = 0
        old_action = -1

        sum_reward = 0
        sum_waiting = 0
        while self.step < self.max_steps:

            # get current state of the intersection
            current_state = self.get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawned in the environment.
            current_total_wait = self.collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self.choose_action(current_state)
            q_values = self.retun_q_values(current_state)

            # record the state action pair in the file
            softmax = torch.nn.Softmax()
            q_list = softmax(q_values[0])
            f.write(str(list(current_state) + q_list.tolist()) + "," + str([action.tolist()]) + "\n")

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self.step != 0 and old_action != action:
                self.set_yellow_phase(old_action)
                self.simulate(self.yellow_duration)

            # execute the phase selected before
            self.set_green_phase(action)
            self.simulate(self.green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                sum_reward += reward
            sum_waiting += current_total_wait

        avg_reward = sum_reward / self.max_steps
        avg_waiting = sum_waiting / self.max_steps
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        f.close()
        return simulation_time, avg_reward, avg_waiting

    def simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        # do not do more steps than the maximum allowed number of steps
        if (self.step + steps_todo) >= self.max_steps:
            steps_todo = self.max_steps - self.step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self.step += 1  # update the step counter
            steps_todo -= 1

    def collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
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

    def choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
        q_value = self.dqn.forward(state)
        return q_value.max(1)[1].data[0]

    def retun_q_values(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
        q_value = self.dqn.forward(state)
        return q_value


    def set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

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
            lane_pos = 750 - lane_pos

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
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
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
                # composition of the two postion ID to create a number in interval 0-79
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




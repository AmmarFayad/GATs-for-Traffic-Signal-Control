import numpy as np
import traci



def get_state_general(self):
        """
        Retrieve the state of multiple intersection from sumo, in the form of cell occupancy
        """
        state = np.array([np.zeros(self.num_states),np.zeros(self.num_states),np.zeros(self.num_states)])
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
                state[0][car_position] = 1

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
                state[1][car_position] = 1

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
                state[2][car_position] = 1
        return state
import numpy as np
import math
import random

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        # how many cars per episode
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        # round every value to int -> effective steps when a car will be generated
        car_gen_steps = np.rint(car_gen_steps)

        """
        Generation of the route of every car for one episode

        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings2 = np.random.weibull(2, self._n_cars_generated)
        timings3 = np.random.exponential(0.1, self._n_cars_generated)
        timings = np.sort(timings)
        timings2 = np.sort(timings2)
        timings3 = np.sort(timings3)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_old2 = math.floor(timings2[1])
        max_old2 = math.ceil(timings2[-1])
        min_old3 = math.floor(timings3[1])
        max_old3 = math.ceil(timings3[-1])
        min_new = 0
        max_new = self._max_steps/3
        max_new2 = self._max_steps*2/3
        max_new3 = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)
        for value in timings2:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new2 - max_new) / (max_old2 - min_old2)) * (value - max_old2) + max_new2)
        for value in timings2:
            car_gen_steps = np.append(car_gen_steps,
                                      ((max_new3 - max_new2) / (max_old3 - min_old3)) * (value - max_old3) + max_new3)

        # round every value to int -> effective steps when a car will be generated
        car_gen_steps = np.rint(car_gen_steps)

        # produce the file for cars generation, one car per line
        """
        d={"A":["B"], "B":["A","B","C"], "C":["B","C","D"], "D":["C","D"]}
        s="""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
            
            """
        #print(d.keys())
        for i in range (15):
            a=random.choice(list(d.keys()))
            #print(a)
            ss=""
            b=str(np.random.randint(1,2))
            for _ in range (np.random.randint(2, 4)):
                
                ss+=a+b
                a=random.choice(list(d[a]))
                if a == ss[-2]:
                    b=str(3-int(ss[-1]))
                    ss+=a+b+" "
                else:
                    b=ss[-1]
                    ss+=a+b+" "
                #print(ss)
            s+="<route id=%s edges=%s/>" % ("\"" + str(i) + "\" ","\"" + ss + "\" ")
            s+='\n'+'\t'
        
        with open("intersection/1intersection/network.rout.xml", "w") as routes:
            print("""<routes>
                    <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
                    
                    <route id="0"  edges="A1B1 B1C1" />
                    <route id="1"  edges="C1B1 B1A1" />
                    <route id="2"  edges="B2B1 B1B0" />
                    <route id="3"  edges="B0B1 B1B2" />
                    
                    
                    	
            """, file=routes)
        # s="<route id="1"  edges="A1B1	B1B2 B2B1" />
        #             	<route id="2"  edges="B1B2	B2B1" />
        #             	<route id="3"  edges="D1C1	C1B1" />
        #             	<route id="4"  edges="B1A1	A1B1" />
        #             	<route id="5"  edges="A1B1	B1B2" />
        #             	<route id="6"  edges="C1C2	C2C1" />
        #             	<route id="7"  edges="A1B1	B1C1" />
        #             	<route id="8"  edges="C1C2	C2C1 C1B1" />
        #             	<route id="9"  edges="A1B1	B1A1" />
        #             	<route id="10"  edges="D1C1	C1D1 D1D2" />
        #             	<route id="11"  edges="B1C1	C1B1 B1A1" />
        #             	<route id="12"  edges="D1C1	C1D1" />
        #             	<route id="13"  edges="D1C1	C1D1" />
        #             	<route id="14"  edges="D1C1	C1B1 B1B2" />
        #                 <route id="15"  edges="E1D1 D1C1 C1C2" />"
        # with open("intersection/episode_routes.rou.xml", "w") as routes:
        #     print("""<routes>
        #     <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

        #     <route id="1" edges="A1B1 B1C1"/>
            
        #     """, file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                # choose direction: straight or turn - 75% of times the car goes straight
                if True:
                    # choose a random source & destination
                    route_straight = np.random.randint(0, 4)
                    
                    print('    <vehicle id="W_E_%i" type="standard_car" route="%s" depart="%s" '
                          'departLane="random" departSpeed="10" />' % (car_counter, route_straight, step), file=routes)
                    
                else:  # 25% of the time, the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" '
                              'departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)

# coding=UTF-8
from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np

"""
Positions:
	location=[-1484.750000, 2039.500000, 62.406250]							# village
	location=[-1590.750000, -162.250000, 54.562500] 						# city
	location=[-2576.500000, 3293.000000, 12.375000]							# tunnel 

	# village to tunnel
	route=[-1484.750000, 2039.500000, 62.406250, 
			-2576.500000, 3293.000000, 12.375000]

	# highway1-village
	location=[-2203.527344, -337.456299, 13.119763]							# south western coast road starting point
	location=[-2816.843994, 2193.854980, 29.460247]							# western coast road middle point, befor the bridge to the tunnle
	location [-722.478088, 5522.071289, 36.470577]
	location=[-437.375763, 5917.432617, 32.426182]
	route=[-2203.527344, -337.456299, 13.119763, 
			-2816.843994, 2193.854980, 29.460247]
	route=[-437.375763, 5917.432617, 32.426182, 	
			-722.478088, 5522.071289, 36.470577, 
			-2816.843994, 2193.854980, 29.460247, 
			-2203.527344, -337.456299, 13.119763]

	# highway2-city
	location = [-1989.000000, -468.250000, 10.562500]						# western coast road
	location = [-1037.957886, -607.505615, 18.152155]						# middle pos
	location = [438.339630, -523.561157, 35.797802]							# middle pos
	location = [1000.059448, -906.553833, 30.398233]						# middle pos
	location = [1053.914917, -1533.013306, 27.520031]						# middle pos
	location = [1171.351563, -1925.791748, 36.220097]						# middle eastern city, industrial zone
	route=[	-1989.000000, -468.250000, 10.562500, 
			-1037.957886, -607.505615, 18.152155,
			438.339630, -523.561157, 35.797802,
			1000.059448, -906.553833, 30.398233,
			1171.351563, -1925.791748, 36.220097
			]

	# city
	location=[-1900.778442, -203.396835, 36.310143]							# western city
	location=[689.279053, 26.910444, 83.943283]								# eastern city
	route=[-1900.778442, -203.396835, 36.310143, 
			689.279053, 26.910444, 83.943283]
"""

FOLLOWING_TRAFFICS = 0x1
YIELD_TO_CROSSING_PEDS = 0x2
DRIVE_AROUND_VEHICLES = 0x4
DRIVE_AROUND_EMPTY_VEHICLES = 0x8

DRIVE_AROUND_PEDS = 0x10
DRIVE_AROUND_OBJECTS = 0x20
UNKOWN_1 = 0x40
STOP_AT_TRAFFIC_LIGHTS = 0x80

USE_BLINKERS = 0x100
ALLOW_GOING_WRONG_WAY = 0x200
GO_IN_REVERSE_GEAR = 0x400  # backwards
UNKOWN_2 = 0x800

UNKOWN_3 = 0x1000
UNKOWN_4 = 0x2000
UNKOWN_5 = 0x4000
UNKOWN_6 = 0x8000

UNKOWN_7 = 0x10000
UNKOWN_8 = 0x20000
TAKE_SHORTEST_PATH = 0x40000  # Removes most pathing limits  the driver even goes on dirtroads. 没有它也能超车
ALLOW_LANE_CHANEG_OVERTAKE = 0x80000

UNKOWN_9 = 0x100000
UNKOWN_10 = 0x200000
IGNORE_ROADS = 0x400000  # Uses local pathing  only works within 200~meters around the player
UNKOWN_11 = 0x800000

IGNORE_ALL_PATHING = 0x1000000  # Goes straight to destination
UNKOWN_12 = 0x2000000
UNKOWN_13 = 0x4000000  # maybe avoid too close laterally
UNKOWN_14 = 0x8000000

UNKOWN_15 = 0x10000000
AVOID_HIGHWAYS_WHEN_POSSIBLE = 0x20000000  # will use the highway if there is no other way to get to the destination
UNKOWN_16 = 0x40000000
UNKOWN_17 = 0x80000000

DEFAULT = DRIVE_AROUND_EMPTY_VEHICLES | DRIVE_AROUND_OBJECTS | USE_BLINKERS | STOP_AT_TRAFFIC_LIGHTS
DUAL_STOP = FOLLOWING_TRAFFICS | YIELD_TO_CROSSING_PEDS
DUAL_AVOID = DRIVE_AROUND_VEHICLES | DRIVE_AROUND_PEDS
LANE_SELECT_1 = 0
LANE_SELECT_2 = ALLOW_LANE_CHANEG_OVERTAKE
LANE_SELECT_3 = ALLOW_LANE_CHANEG_OVERTAKE | TAKE_SHORTEST_PATH | ALLOW_GOING_WRONG_WAY

STRICT_1 = DEFAULT | DUAL_STOP | LANE_SELECT_1
STRICT_2 = DEFAULT | DUAL_STOP | LANE_SELECT_2
LOOSE_1 = DEFAULT | DUAL_STOP | DUAL_AVOID | LANE_SELECT_2
LOOSE_2 = DEFAULT | DUAL_STOP | DUAL_AVOID | LANE_SELECT_3
LOOSE_3 = DEFAULT | DUAL_AVOID | LANE_SELECT_3
AVOID_PED = DEFAULT | DUAL_AVOID | LANE_SELECT_3

# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    parser.add_argument('-d', '--dataset_path', default=os.path.join(os.getcwd(), 'data/'),
                        help='Place to store the dataset')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port
    client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9, divideByTrip=True)

    dataset = Dataset(
        rate=10, frame=[480, 320], throttle=True, brake=True, steering=True,
        speed=True, acceleration=True, yaw=True, yawRate=True, isCollide=True,
        location=True, drivingModeMsg=True,
        lidar=[3, True, 100.0, 1000, 60.0, 300.0, 20, 85.0, 115.0],
        vehicles=True, peds=True
    )
    # Automatic driving scenario
    scenario = Scenario(vehicle='blista', time=[12, 0], drivingMode=[-2, 3, 25.0, 1.0, 1.0],
                        route=[
                            -1989.000000, -468.250000, 10.562500,
                            689.279053, 26.910444, 83.943283
                        ])

    client.send_message(Start(scenario=scenario, dataset=dataset))  # Start request
    count = 0
    tripNum = 0
    old_location = [0, 0, 0]
    while True:  # Main loop
        try:
            # Message recieved as a Python dictionary
            message = client.recv_message()

            if (count % 500) == 0:
                print('loop count is:', count)

            if 'drivingMode' in message:
                if message['drivingMode'][0] == 0:
                    print('trip' + str(tripNum))
                    tripNum += 1

            count += 1

        except KeyboardInterrupt:
            i = input('Paused. Press p to continue and q to exit... ')
            if i == 'p':
                continue
            elif i == 'q':
                break

    # DeepGTAV stop message
    client.send_message(Stop())
    client.close()

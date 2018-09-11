#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from ctypes import *
from numpy.lib.stride_tricks import as_strided


class Scenario:
    """
    Scenario args:
	surroundDrivingMode args:
		args0: int params flag:
			args0 == -2 use default GTAV AI driver, args1~3 are useless
			args0 == -1 use preinstall driving style as the surrounding driving mode, args2~3 are useless
			args0 >=  0 use manual setting driving style. And this args is the driving style int
		args1: an args depends on args0
			if args0 == -2, it's useless
			if args0 == -1, it's the preinstall driving style index
			if args0 >=  0, it's the desired speed
		args2: only activated if args0 >= 0
			aggressiveness
		args3: only activated if args0 >= 0
			ability
		example:
			use default GTAV AI driver:
				[-2]
			use the 2rd(begins from 0) preinstall driving style:
				[-1, 1]
			use manual driving style, set desired speed=30.0, aggressiveness=0.0, ability=1.0:
				[RUSHED, 30.0, 0.0, 1.0]

	drivingMode args:
		args0: int, ego driving style index
			args0 == -2	Manual driving:	just drive, args1~args4 are useless
			args0 == -1	Auto driving:	auto selecting preinstall driving styles, desired speed. manual setting route mode by turns when the vehicle reach the destination. Args2~args4 are useless
			args0 >=  0	Auto driving:	manual setting  driving styles, desired speed, driving aggressiveness, driving ability and route mode
		args1: if args0 >= -1, int, ego route mode, WANDERING==0, TO_COORD_ONE_WAY_TRIP==1, TO_COORD_CIRCLE_TRIP==2, TO_COORD_ONE_WAY_TRIP_CIRCLE==3
		args2: if args0 >=  0, float, ego desired speed
		args3: if args0 >=  0, float, ego driving aggressiveness of driver
		args4: if args0 >=  0, float, ego driving ability of driver
		example:
			use manual driving:
				[-2] or None
			use preinstall driving style:
				[-1]
			use manual driving style, set driving style int=3, TO_COORD_ONE_WAY_TRIP_CIRCLE, desired speed=30.0, aggressiveness=1.0, ability=1.0:
				[3, 2, 30, 1.0, 1.0]

	route args:
		args0~args2: float, start position XYZ
		args3~args5: float, destination or middle position
		...
		argsn~argsn+2: float, destination
		example:
			route=[-1590.750000, -162.250000, 54.562500, -1592.500000, -197.500000, 54.281250]
			test loop: [-1590.750000, -162.250000, 54.562500] to [-1592.500000, -197.500000, 54.281250]

    """

    def __init__(self, location=None, time=None, weather=None, vehicle=None, drivingMode=None, route=None,
                 surroundDrivingMode=None):
        self.location = location  # [x,y]
        self.time = time  # [hour, minute]
        self.weather = weather  # string
        self.vehicle = vehicle  # string
        self.drivingMode = drivingMode
        self.route = route
        self.surroundDrivingMode = surroundDrivingMode


class Dataset:
    """
    Datasets:
	lidar args, average samples:
		args0: int, lidar state flag:
			LIDAR_NOT_INIT_YET,
			LIDAR_INIT_AS_2D,
			LIDAR_INIT_AS_3D_CONE,
			LIDAR_INIT_AS_3D_SCALED_CONE,
			LIDAR_INIT_AS_3D_SPACIALCIRCLE,
			LIDAR_INIT_AS_3D_SCALED_SPACIALCIRCLE
		args1: bool, visualize the laser dots
		args2: float, lidar laser max range
		args3: int, horizontal sample number
		args4: float, horizontal angle left limit, degree
		args5: float, horizontal angle right limit, degree
		args6: int, vertical sample number
		args7: float, vertical upper limit, degree
		args8: float, vertical under limit, degree

		example:
			lidar=[1, False, 100.0, 1080, 90.0, 270.0]	# 2D lidar
			lidar=[2, False, 100.0, 180, 90.0, 270.0, 15, 85.0, 130.0]	# 3D lidar

	lidar args, scaled samples, 3D lidar only:
		args0: int, lidar state flag, 3-LIDAR_INIT_AS_3D_SCALED, LIDAR_INIT_AS_3D_SCALED_SPACIALCIRCLE
		args1: bool, visualize the laser dots
		args2: float, lidar laser max range
		args3: int, total sample number
		args4: float, horizontal angle left limit, degree
		args5: float, horizontal angle right limit, degree
		args6: int, vertical sample number
		args7: float, vertical upper limit, degree
		args8: float, vertical under limit, degree

		example:
			lidar=[3, False, 100.0, 1200, 60.0, 300.0, 20, 85.0, 115.0]

    """

    def __init__(self, rate=None, frame=None, vehicles=None, peds=None, trafficSigns=None, direction=None,
                 reward=None, throttle=None, brake=None, steering=None, speed=None, yaw=None, yawRate=None,
                 drivingModeMsg=None,
                 location=None, time=None, rageMatrices=None, cameraInfo=None, eulerAngles=None, lidar=None,
                 isCollide=None, acceleration=None):
        self.rate = rate  # Hz
        self.frame = frame  # [width, height]
        self.vehicles = vehicles  # boolean
        self.peds = peds  # boolean
        self.trafficSigns = trafficSigns  # boolean
        self.direction = direction  # [x,y,z]
        self.reward = reward  # [id, p1, p2]
        self.throttle = throttle  # boolean
        self.brake = brake  # boolean
        self.steering = steering  # boolean
        self.speed = speed  # boolean
        self.yaw = yaw  # boolean
        self.yawRate = yawRate  # boolean
        self.drivingModeMsg = drivingModeMsg  # boolean
        self.location = location  # boolean
        self.time = time  # boolean
        self.rageMatrices = rageMatrices  # boolean
        self.cameraInfo = cameraInfo  # boolean
        self.eulerAngles = eulerAngles  # boolean
        self.lidar = lidar  # boolean
        self.isCollide = isCollide  # boolean
        self.acceleration = acceleration  # boolean


class Start:
    def __init__(self, scenario=None, dataset=None):
        self.scenario = scenario
        self.dataset = dataset

    def to_json(self):
        _scenario = None
        _dataset = None

        if (self.scenario != None):
            _scenario = self.scenario.__dict__

        if (self.dataset != None):
            _dataset = self.dataset.__dict__

        return json.dumps({'start': {'scenario': _scenario, 'dataset': _dataset}})

    def activate_bytes_frame(self):
        return True if self.dataset.frame != None else False, True if self.dataset.lidar != None else False


class Config:
    def __init__(self, scenario=None, dataset=None):
        self.scenario = scenario
        self.dataset = dataset

    def to_json(self):
        _scenario = None
        _dataset = None

        if (self.scenario != None):
            _scenario = self.scenario.__dict__

        if (self.dataset != None):
            _dataset = self.dataset.__dict__

        return json.dumps({'config': {'scenario': _scenario, 'dataset': _dataset}})

    def activate_bytes_frame(self):
        return True if self.dataset.frame != None else False, True if self.dataset.lidar != None else False


class Stop:
    def to_json(self):
        return json.dumps({'stop': None})  # super dummy


class Commands:
    def __init__(self, throttle=None, brake=None, steering=None):
        self.throttle = throttle  # float (0,1)
        self.brake = brake  # float (0,1)
        self.steering = steering  # float (-1,1)

    def to_json(self):
        return json.dumps({'commands': self.__dict__})


class Ray(Structure):
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float),
        ('entityType', c_int),
        ('rayResult', c_int),
        ('range', c_float)
    ]


def frame2numpy(frame, frameSize):
    buff = np.fromstring(frame, dtype='uint8')
    # Scanlines are aligned to 4 bytes in Windows bitmaps
    strideWidth = int((frameSize[0] * 3 + 3) / 4) * 4
    # Return a copy because custom strides are not supported by OpenCV.
    return as_strided(buff, strides=(strideWidth, 3, 1), shape=(frameSize[1], frameSize[0], 3)).copy()

def lidar2numpy(lidar):
    result = []
    x = Ray()
    ##TODO: parse the lidar data with struct Ray
    records = iter(lidar, sizeof(x))
    for item in records:
        lidar_dict = {}
        lidar_dict['x'] = item.x
        lidar_dict['y'] = item.y
        lidar_dict['z'] = item.z
        lidar_dict['entityType'] = item.entityType
        lidar_dict['rayResult'] = item.rayResult
        lidar_dict['range'] = item.range
        result.append(lidar_dict)
    return result

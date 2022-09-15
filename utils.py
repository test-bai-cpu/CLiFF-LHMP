import json
from enum import Enum
from math import dist

import numpy as np
import pandas as pd



class Method(Enum):
    MoD = 1
    CVM = 2

class Dataset(Enum):
    ATC = 1
    THOR1 = 2
    THOR3 = 3

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def millimeter_to_meter(data, column_names):
    for column_name in column_names:
        data[column_name] = data[column_name].apply(lambda x: x / 1000)
    return data


def get_euclidean_distance(position_array_1, position_array_2):
    return dist(position_array_1, position_array_2)


def get_euclidean_distance_point(x1, y1, x2, y2):
    return dist((x1, y1), (x2, y2))


def get_mahalanobis_distance(point, SWGMM):
    mean = SWGMM[2:4]
    cov = [SWGMM[4:6], SWGMM[6:8]]
    mahalanobis_distance = np.dot(np.dot((point - mean).T, np.linalg.inv(cov)), (point - mean))

    return mahalanobis_distance

### Functions for reading data
# atc
def read_human_traj_data_by_person_id(human_traj_file, person_id):
    data = pd.read_csv(human_traj_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "z", "velocity", "motion_angle", "facing_angle"]
    data = millimeter_to_meter(data, ["x", "y", "z", "velocity"])
    human_traj_data = data.loc[data['person_id'] == person_id]
    human_traj_array = human_traj_data[["time", "x", "y", "velocity", "motion_angle"]].to_numpy()

    return human_traj_array

# atc
def read_human_traj_data(human_traj_file):
    data = pd.read_csv(human_traj_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "z", "velocity", "motion_angle", "facing_angle"]
    data = millimeter_to_meter(data, ["x", "y", "z", "velocity"])

    return data

# thor
def read_thor_human_traj_data(human_traj_file):
    data = pd.read_csv(human_traj_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
    return data

def read_cliff_map_data(cliff_map_file):
    data = pd.read_csv(cliff_map_file, header=None)
    data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight",
                    "motion_ratio", "observation_ratio"]

    return data.to_numpy()

### Functions for reading maps

def read_pgm(pgmf):
    with open(pgmf) as file:
        pgm_header = []
        for i in range(3):
            pgm_header.append(file.readline())
        
        (width, height) = [int(i) for i in pgm_header[2].split()]
        matrix = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(file.read(1)))
            matrix.append(row)
        return pgm_header, matrix

def check_if_in_occupied_area(image_array, new_position):
    # Map is 168m * 72m, (0,0) is the center. Pgm is 2800 * 1200 pixel
    # new_position = np.array([next_position_x, next_position_y])
    x = new_position[0]
    y = new_position[1]
    # pixel_x = round((x + 45) * 1900 / 95 + 950)
    pixel_x = round((x + 60) * 2800 / 140)
    pixel_y = round((y + 40) * 1200 / 60)
    grey_value = image_array[1200 - pixel_y][pixel_x]
    if grey_value == 127:
        return False
    return True

def read_atc_map():
    # map1 is precise, map4 is larger
    _, raw_image_data = read_pgm("atc-map/map4.pgm")
    image_array = np.array(raw_image_data)
    return image_array

def save_data_list(data_list, file_name):
    with open(file_name, "w") as f:
        f.write(json.dumps(data_list))

def read_data_list(file_name):
    with open(file_name, "r") as f:
        data_list = json.loads(f.read())
    
    return data_list

def _circfuncs_common(samples, high, low):
    # Ensure samples are array-like and size is not zero
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low)*2.* np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.* np.pi / (high - low))

    return samples, sin_samp, cos_samp

def circmean(samples, weights, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sum(sin_samp * weights)
    cos_sum = sum(cos_samp * weights)
    res = np.arctan2(sin_sum, cos_sum)
    res = res*(high - low)/2.0/np.pi + low
    return wrapTo2pi(res)

def circdiff(circular_1, circular_2):
    res = np.arctan2(np.sin(circular_1-circular_2), np.cos(circular_1-circular_2))
    return abs(res)

# atc -pi,pi
# cliff 0, 2pi
def wrapTo2pi(circular_value):
    return np.mod(circular_value,2*np.pi)
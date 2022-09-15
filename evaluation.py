import numpy as np

import utils


def get_position_in_predicted_traj(timestamp, predicted_traj):
    if timestamp > predicted_traj[-1,0]:
        return None
    diff_array = np.abs(predicted_traj[:, 0] - timestamp)
    min_time_diff_index = np.argmin(diff_array)
    nearest_time = predicted_traj[min_time_diff_index][0]
    if diff_array[min_time_diff_index] == 0:
        return predicted_traj[min_time_diff_index][1:3]
    if timestamp > nearest_time:
        i1 = min_time_diff_index
        i2 = min_time_diff_index + 1
    if timestamp < nearest_time:
        i1 = min_time_diff_index - 1
        i2 = min_time_diff_index

    proportion = (timestamp - predicted_traj[i1][0]) / (predicted_traj[i2][0] - predicted_traj[i1][0])
    x = proportion * (predicted_traj[i2][1] - predicted_traj[i1][1]) + predicted_traj[i1][1]
    y = proportion * (predicted_traj[i2][2] - predicted_traj[i1][2]) + predicted_traj[i1][2]

    return np.array([x, y])


def get_position_in_ground_truth(timestamp, traj):
    if timestamp > traj[-1,0]:
        return None
    diff_array = np.abs(traj[:, 0] - timestamp)
    min_time_diff_index = np.argmin(diff_array)
    nearest_time = traj[min_time_diff_index][0]
    if diff_array[min_time_diff_index] == 0:
        return traj[min_time_diff_index][1:3]
    if timestamp > nearest_time:
        i1 = min_time_diff_index
        i2 = min_time_diff_index + 1
    if timestamp < nearest_time:
        i1 = min_time_diff_index - 1
        i2 = min_time_diff_index

    proportion = (timestamp - traj[i1][0]) / (traj[i2][0] - traj[i1][0])
    x = proportion * (traj[i2][1] - traj[i1][1]) + traj[i1][1]
    y = proportion * (traj[i2][2] - traj[i1][2]) + traj[i1][2]

    return np.array([x, y])


def get_ADE_with_ground_truth_timestamp(ground_truth_traj, predicted_traj):
    ADE = []
    for ground_truth_point in ground_truth_traj:
        timestamp = ground_truth_point[0]
        ground_truth_position = ground_truth_point[1:3]
        predicted_position = get_position_in_predicted_traj(timestamp, predicted_traj)
        if predicted_position is None:
            print("Ground truth traj is longer than prediction traj. ADE ends here.")
            break
        ADE.append(utils.get_euclidean_distance(ground_truth_position, predicted_position))

    ADE_array = np.array(ADE)
    ADE_mean = np.mean(ADE_array)

    return ADE_mean


def get_ADE_with_predict_timestamp(ground_truth_traj, predicted_traj, start_predict_position):
    ADE = []
    for predicted_point in predicted_traj[start_predict_position+1:]:
        timestamp = predicted_point[0]
        predicted_position = predicted_point[1:3]
        ground_truth_position = get_position_in_ground_truth(timestamp, ground_truth_traj)
        if ground_truth_position is None:
            print("Prediction traj is longer than ground truth traj. ADE ends here.")
            break
        ADE.append(utils.get_euclidean_distance(ground_truth_position, predicted_position))

    ADE_array = np.array(ADE)
    ADE_mean = np.mean(ADE_array)

    return ADE_mean, ADE


def get_error_list_with_predict_timestamp(ground_truth_traj, predicted_traj, start_predict_position):
    ADE = []
    for predicted_point in predicted_traj[start_predict_position+1:]:
        timestamp = predicted_point[0]
        predicted_position = predicted_point[1:3]
        ground_truth_position = get_position_in_ground_truth(timestamp, ground_truth_traj)
        if ground_truth_position is None:
            print("Prediction traj is longer than ground truth traj. ADE ends here.")
            break
        ADE.append(utils.get_euclidean_distance(ground_truth_position, predicted_position))

    return ADE


def get_FDE_with_predict_timestamp(ground_truth_traj, predicted_traj):
    last_prediction_point = predicted_traj[-1, :]
    timestamp = last_prediction_point[0]
    last_prediction_position = last_prediction_point[1:3]
    ground_truth_position = get_position_in_ground_truth(timestamp, ground_truth_traj)
    if ground_truth_position is None:
        print("Prediction traj is longer than ground truth traj. FDE ends here.")
        return None
    FDE = utils.get_euclidean_distance(ground_truth_position, last_prediction_position)

    return FDE


def get_FDE_with_ground_truth_timestamp(ground_truth_traj, predicted_traj):
    last_ground_truth_point = ground_truth_traj[-1, :]
    timestamp = last_ground_truth_point[0]
    last_ground_truth_position = last_ground_truth_point[1:3]
    last_prediction_position = get_position_in_predicted_traj(timestamp, predicted_traj)
    if last_prediction_position is None:
        FDE = get_FDE_with_predict_timestamp(ground_truth_traj, predicted_traj)
    else:
        FDE = utils.get_euclidean_distance(last_ground_truth_position, last_prediction_position)

    return FDE

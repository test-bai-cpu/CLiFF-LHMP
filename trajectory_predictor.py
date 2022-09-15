import math
import csv
from random import betavariate

import numpy as np

import utils
import evaluation


class TrajectoryPredictor:
    def __init__(
            self,
            *,
            cliff_map_origin_data,
            human_traj_origin_data,
            person_id: int = None,
            start_length: int = 0,
            observed_tracklet_length: int = 1,
            max_planning_horizon: int = 50,
            beta: float = 1,
            sample_radius: float = 1,
            delta_t: int = 1,
            result_file: str,
    ):
        self.cliff_map = cliff_map_origin_data
        self.human_traj_data = human_traj_origin_data
        self.person_id = person_id
        self.start_length = start_length
        self.observed_tracklet_length = observed_tracklet_length
        self.max_planning_horizon = max_planning_horizon
        self.planning_horizon = None
        self.beta = beta
        self.sample_radius = sample_radius
        self.delta_t = delta_t
        self.result_file = result_file

    def set_planning_horizon(self):
        ground_truth_time = math.floor(self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0])
        self.planning_horizon = min(ground_truth_time, self.max_planning_horizon)

    def check_human_traj_data(self):
        if self.person_id == -1:
            return False
        row_num = self.human_traj_data.shape[0]
        if row_num <= self.start_length + self.observed_tracklet_length + 1:
            return False
        if (self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0]) < self.delta_t:
            return False

        idle_threshold = 1
        idle_check_interval = 10
        distance_array = self.human_traj_data[:-idle_check_interval, 1:3] - self.human_traj_data[idle_check_interval:, 1:3]
        distance = np.sqrt(np.power(distance_array[:,0], 2) + np.power(distance_array[:,1], 2))
        if np.any(distance < idle_threshold):
            return False

        self.set_planning_horizon()
        return True

    def predict_one_human_traj_mod(self):
        total_predicted_motion_list = []
        current_motion_origin = self._calculate_current_motion()

        for _ in range(20):
            current_motion = current_motion_origin
            predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
            for time_index in range(1, int(self.planning_horizon / self.delta_t) + 2):
                SWGMM_in_cliffmap = self._find_near_high_observed_SWGMM_in_cliffmap(current_motion)
                if not SWGMM_in_cliffmap:
                    break
                else:
                    sampled_velocity = self._sample_motion_from_SWGMM_in_cliffmap(SWGMM_in_cliffmap)
                    updated_motion = self._apply_sampled_motion_to_current_motion(sampled_velocity, current_motion)
                predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
                current_motion = self._predict_with_constant_velocity_model(updated_motion)

            total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list

    def predict_one_human_traj_cvm(self):
        total_predicted_motion_list = []
        current_motion_origin = self._calculate_current_motion()

        current_motion = current_motion_origin
        predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
        for time_index in range(1, int(self.planning_horizon / self.delta_t) + 2):
            updated_motion = current_motion
            predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
            current_motion = self._predict_with_constant_velocity_model(updated_motion)

        total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list

    def evaluate_ADE_FDE_result(self, all_predicted_trajectory_list):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append([round(num, 3) for num in error_list])

        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            for error_list in error_matrix:
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            FDE_mean = round(np.mean(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_min = round(np.min(traj_error_array_for_one_time_index[:,-1]), 3)
            ADE_list = np.mean(traj_error_array_for_one_time_index, axis=1)
            ADE_mean = round(np.mean(ADE_list), 3)
            ADE_min = round(np.min(ADE_list), 3)
            top_k_ADE_by_timestep = round(np.mean(np.min(traj_error_array_for_one_time_index, axis=0)), 3)
            data_row = [self.person_id, round(time_index*self.delta_t, 1), FDE_mean, FDE_min, ADE_mean, ADE_min, top_k_ADE_by_timestep, num_predicted_trajs]

            if time_index >= 20:
                res_FDE = FDE_mean
                res_ADE = ADE_mean
            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)

        return res_FDE, res_ADE

    def _calculate_current_motion(self):
        current_motion_origin = self.human_traj_data[self.start_length + self.observed_tracklet_length, :]
        sigma = 1.5
        current_speed = 0
        current_orientation = 0
        g_t = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- t ** 2 / (2 * sigma ** 2)) for t in range(1, self.observed_tracklet_length + 1)]
        g_t = [g/sum(g_t) for g in g_t]
        g_t = np.flip(g_t)
        raw_speed_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 3]
        raw_orientation_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 4]
        weighted_speed_list = raw_speed_list * g_t
        current_speed = np.sum(weighted_speed_list)
        wrapped_orientation = utils.wrapTo2pi(raw_orientation_list)
        current_orientation = utils.circmean(wrapped_orientation, g_t)
        current_motion = np.concatenate((current_motion_origin[0:3], [current_speed, current_orientation]))
        return current_motion

    def _predict_with_constant_velocity_model(self, updated_motion):
        new_position = self._get_next_position_by_velocity(updated_motion[1:3], updated_motion[3:5])
        new_timestamp = np.array([updated_motion[0] + self.delta_t])
        predicted_motion = np.concatenate((new_timestamp, new_position, updated_motion[3:5]))
        return predicted_motion

    def _sample_motion_from_SWGMM_in_cliffmap(self, SWGMM_in_cliffmap):
        SWND = self._sample_component_from_SWGMM(SWGMM_in_cliffmap)

        mean = SWND[2:4]
        cov = [SWND[4:6], SWND[6:8]]

        sampled_velocity = np.random.multivariate_normal(mean, cov, 1)
        samapled_rho = sampled_velocity[0][1]
        sampled_theta = utils.wrapTo2pi(sampled_velocity[0][0])
        sampled_velocity_rho_theta = np.array([samapled_rho, sampled_theta])

        return sampled_velocity_rho_theta

    def _sample_component_from_SWGMM(self, SWGMM_in_cliffmap):
        component_weight_list = []
        for SWND in SWGMM_in_cliffmap:
            component_weight_list.append(SWND[8])

        component_weight_array = np.array(component_weight_list)
        component_weight_normalize = list(component_weight_array / component_weight_array.sum())

        component_weight_acc = [np.sum(component_weight_normalize[:i]) for i in range(1, len(component_weight_normalize)+1)]
        r = np.random.uniform(0, 1)
        index = 0
        for i, threshold in enumerate(component_weight_acc):
            if r < threshold:
                index = i
                break
        
        SWND = SWGMM_in_cliffmap[index]

        return SWND

    def _apply_sampled_motion_to_current_motion(self, sampled_velocity, current_motion):
        current_velocity = current_motion[3:5]
        result_rho = current_velocity[0]
        sampled_orientation = utils.wrapTo2pi(sampled_velocity[1])
        current_orientation = utils.wrapTo2pi(current_velocity[1])
        delta_theta = utils.circdiff(sampled_orientation, current_orientation)
        param_lambda = self._apply_gaussian_kernel(delta_theta)
        result_theta = utils.circmean([sampled_orientation, current_orientation], [param_lambda, 1-param_lambda])
        predicted_motion = np.concatenate(
            (current_motion[0:3], [result_rho, result_theta])
        )

        return predicted_motion

    def _apply_gaussian_kernel(self, x):
        return np.exp(-self.beta*x**2)

    def _find_near_high_observed_SWGMM_in_cliffmap(self, current_motion):
        near_SWGMM = []
        current_location = current_motion[1:3]
        location_array = self.cliff_map[:,0:2]
        near_index_list = np.where(np.sum(np.power(location_array - current_location, 2), axis=1) < self.sample_radius)
        max_observation_ratio = None
        max_observation_ratio_index = None
        if len(near_index_list[0]) == 0:
            return None
        for index in near_index_list[0]:
            observation_ratio = self.cliff_map[index, 10]
            if (not max_observation_ratio) or observation_ratio > max_observation_ratio:
                max_observation_ratio = observation_ratio
                max_observation_ratio_index = index
        
        index_list = np.where((location_array[:, 0] == location_array[max_observation_ratio_index][0]) & (location_array[:, 1] == location_array[max_observation_ratio_index][1]))
        for index in index_list[0]:
            near_SWGMM.append(self.cliff_map[index].tolist())

        return near_SWGMM

    def _get_next_position_by_velocity(self, current_position, current_velocity):
        next_position_x = current_position[0] + current_velocity[0] * np.cos(current_velocity[1]) * self.delta_t
        next_position_y = current_position[1] + current_velocity[0] * np.sin(current_velocity[1]) * self.delta_t

        return np.array([next_position_x, next_position_y])

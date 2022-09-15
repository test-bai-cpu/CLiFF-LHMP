import matplotlib.pyplot as plt
from tqdm import *
import csv

from trajectory_predictor import TrajectoryPredictor
import plot_figures
import utils


def get_all_person_id(data):
    person_id_list = list(data.person_id.unique())
    return person_id_list


def get_human_traj_data_by_person_id(human_traj_origin_data, person_id):
    human_traj_data_by_person_id = human_traj_origin_data.loc[human_traj_origin_data['person_id'] == person_id]
    human_traj_array = human_traj_data_by_person_id[["time", "x", "y", "velocity", "motion_angle"]].to_numpy()

    return human_traj_array


def run_experiment(dataset, map_file, mod_file, ground_truth_data_file, result_file, observed_tracklet_length, start_length, planning_horizon, beta, sample_radius, delta_t, method):
    cliff_map_data = utils.read_cliff_map_data(mod_file)

    if dataset == utils.Dataset.ATC:
        fig_size = [-60, 80, -40, 20]
        human_traj_data = utils.read_human_traj_data(ground_truth_data_file)
    elif dataset in [utils.Dataset.THOR1, utils.Dataset.THOR3]:
        fig_size = [-11, 12, -10, 13.5]
        human_traj_data = utils.read_thor_human_traj_data(ground_truth_data_file)
    
    plt.subplot(111, facecolor='grey')
    img = plt.imread(map_file)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=fig_size)
    plot_figures.plot_cliff_map(cliff_map_data)

    person_id_list = get_all_person_id(human_traj_data)
    print("In total, there are " + str(len(person_id_list)) + " trajectories.")

    header = ["person_id", "predict_horizon", "FDE_mean", "FDE_min", "ADE_mean", "ADE_min", "num_predicted_trajs"]
    with open(result_file, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    # Example person ids for testing
    # person_id_list = [13185100] # ATC
    # person_id_list = [12063] # THÖR1
    # person_id_list = [32096] # THÖR3

    for person_id in tqdm(person_id_list):
        human_traj_data_by_person_id = get_human_traj_data_by_person_id(human_traj_data, person_id)
        trajectory_predictor = TrajectoryPredictor(
            cliff_map_origin_data=cliff_map_data,
            human_traj_origin_data=human_traj_data_by_person_id,
            person_id=person_id,
            start_length=start_length,
            observed_tracklet_length=observed_tracklet_length,
            max_planning_horizon=planning_horizon,
            beta=beta,
            sample_radius=sample_radius,
            delta_t=delta_t,
            result_file=result_file,
        )
        if not trajectory_predictor.check_human_traj_data():
            continue

        if method == utils.Method.MoD:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_mod()
        elif method == utils.Method.CVM:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_cvm()

        trajectory_predictor.evaluate_ADE_FDE_result(all_predicted_trajectory_list)

        plot_figures.plot_all_predicted_trajs(all_predicted_trajectory_list, observed_tracklet_length)
        plot_figures.plot_human_traj(human_traj_data_by_person_id[start_length:, :], observed_tracklet_length)

    plt.show()


def main_atc():
    map_file = "maps/atc.jpg"
    mod_file = "MoDs/atc/atc-20121024-cliff.csv"
    ground_truth_data_file = "dataset/atc/atc-20121031_2.5_ds.csv"
    result_file = "atc_results.csv"
    observed_tracklet_length = 8
    start_length = 0
    planning_horizon = 50
    beta = 1
    sample_radius = 1
    delta_t = 1
    method = utils.Method.CVM
    dataset = utils.Dataset.ATC
    run_experiment(dataset, map_file, mod_file, ground_truth_data_file, result_file, observed_tracklet_length, start_length, planning_horizon, beta, sample_radius, delta_t, method)

def main_thor1():
    map_file = "maps/thor1.png"
    mod_file = "MoDs/thor1/thor1_cliff.csv"
    ground_truth_data_file = "dataset/thor1/exp-1-run-2.csv"
    result_file = "thor1_results.csv"
    observed_tracklet_length = 8
    start_length = 0
    planning_horizon = 12
    beta = 1
    sample_radius = 0.5
    delta_t = 0.4
    method = utils.Method.MoD
    dataset = utils.Dataset.THOR1
    run_experiment(dataset, map_file, mod_file, ground_truth_data_file, result_file, observed_tracklet_length, start_length, planning_horizon, beta, sample_radius, delta_t, method)

def main_thor3():
    map_file = "maps/thor3.png"
    mod_file = "MoDs/thor3/thor3_cliff.csv"
    ground_truth_data_file = "dataset/thor3/exp-3-run-2.csv"
    result_file = "thor3_results.csv"
    observed_tracklet_length = 8
    start_length = 0
    planning_horizon = 12
    beta = 1
    sample_radius = 0.5
    delta_t = 0.4
    method = utils.Method.MoD
    dataset = utils.Dataset.THOR3
    run_experiment(dataset, map_file, mod_file, ground_truth_data_file, result_file, observed_tracklet_length, start_length, planning_horizon, beta, sample_radius, delta_t, method)

if __name__ == "__main__":
    main_thor3()
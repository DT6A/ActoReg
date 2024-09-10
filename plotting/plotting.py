import os
import pickle
import random
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


os.makedirs("out", exist_ok=True)
os.makedirs("bin", exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
sns.set(style="ticks", font_scale=1.5)
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.serif': 'Times New Roman'
})

# # EOP

"""
Taken from here and a bit modified
https://github.com/dodgejesse/show_your_work/
"""


def _cdf_with_replacement(i, n, N):
    return (i / N) ** n


def _compute_stds(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    std_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_std = 0
        for i in range(N):
            cur_std += (cur_data[i] - expected_max_cond_n[n]) ** 2 * pdfs[n][i]
        cur_std = np.sqrt(cur_std)
        std_of_max_cond_n.append(cur_std)
    return std_of_max_cond_n


# this implementation assumes sampling with replacement for computing the empirical cdf
def expected_online_performance(
        online_performance: List[float],
        output_n: int
) -> Dict[str, Union[List[float], float]]:
    # Copy and sort?
    online_performance = list(online_performance)
    online_performance.sort()

    N = len(online_performance)
    pdfs = []
    for n in range(1, N + 1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1, N + 1):
            F_Y_of_y.append(_cdf_with_replacement(i, n, N))

        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]

        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    expected_med_cond_n = []
    expected_iqr_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += online_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)

        # estimate median
        cur_sum = 0.0
        for i in range(N):
            cur_sum += pdfs[n][i]
            if cur_sum == 0.5:
                expected_med_cond_n.append(online_performance[i])
                break
            elif cur_sum > 0.5:
                # nearest strat
                cur_diff = cur_sum - 0.5
                prev_dif = 0.5 - (cur_sum - pdfs[n][-1])
                if cur_diff < prev_dif:
                    expected_med_cond_n.append(online_performance[i])
                else:
                    expected_med_cond_n.append(online_performance[i - 1])
                break

        # estimate iqr
        cur_sum = 0.0
        percent25 = 0.0
        checked25 = False

        percent75 = 0.0
        checked75 = False
        for i in range(N):
            cur_sum += pdfs[n][i]
            if not checked25:
                if cur_sum == 0.25:
                    percent25 = online_performance[i]
                    checked25 = True
                elif cur_sum > 0.25:
                    # nearest strat
                    cur_diff = cur_sum - 0.25
                    prev_dif = 0.25 - (cur_sum - pdfs[n][-1])
                    if cur_diff < prev_dif:
                        percent25 = online_performance[i]
                    else:
                        percent25 = online_performance[i - 1]

            if not checked75:
                if cur_sum == 0.75:
                    percent75 = online_performance[i]
                    checked75 = True
                elif cur_sum > 0.75:
                    # nearest strat
                    cur_diff = cur_sum - 0.75
                    prev_dif = 0.75 - (cur_sum - pdfs[n][-1])
                    if cur_diff < prev_dif:
                        percent75 = online_performance[i]
                    else:
                        percent75 = online_performance[i - 1]
        expected_iqr_cond_n.append(percent75 - percent25)

    std_of_max_cond_n = _compute_stds(N, online_performance, expected_max_cond_n, pdfs)

    return {
        "median": expected_med_cond_n[:output_n],
        "iqr": expected_iqr_cond_n[:output_n],
        "mean": expected_max_cond_n[:output_n],
        "std": std_of_max_cond_n[:output_n],
        "max": np.max(online_performance),
        "min": np.min(online_performance)
    }


def expected_online_performance_arbit(
        online_performance: List[float],
        offline_performance: List[float],
        output_n: int
) -> Dict[str, Union[List[float], float]]:
    means = [x for _, x in sorted(zip(offline_performance, online_performance), key=lambda pair: pair[0], reverse=True)]

    if len(means) > 0:
        cur_max = means[0]
        for ind in range(len(means)):
            cur_max = max(cur_max, means[ind])
            means[ind] = cur_max

    return {
        "mean": means[:output_n],
        "std": means[:output_n],
        "max": np.max(online_performance),
        "min": np.min(online_performance)
    }


def get_data_from_sweeps(sweeps_ids, param_1="actor_bc_coef", param_2="critic_bc_coef", param_3=None, only_last=True,
                         params_filter=dict()):
    maxes = []
    lasts = []
    name_list = []
    config_list = []
    full_scores = {}

    for s in tqdm(sweeps_ids, desc="Sweeps processing", position=0, leave=True):
        api = wandb.Api(timeout=39)
        sweep = api.sweep(s)
        runs = sweep.runs
        cur_max = 0
        for run in tqdm(runs, desc="Runs processing", position=0, leave=True):
            all_scores = []

            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            skip = False
            for fk in params_filter:
                if config[fk] != params_filter[fk]:
                    skip = True
                    break
            if skip:
                continue
            # print(run.name, end=' ')
            for i, row in run.history(keys=["eval/normalized_score_mean"], samples=3000).iterrows():
                last = row["eval/normalized_score_mean"]
                all_scores.append(last)
            cur_max = max(cur_max, len(all_scores))
            if len(all_scores) < 100 and "antmaze" not in config["dataset_name"]:
                all_scores = [0] * cur_max
            if config["dataset_name"] not in full_scores:
                full_scores[config["dataset_name"]] = {}
            if str(config[param_1]) not in full_scores[config["dataset_name"]]:
                full_scores[config["dataset_name"]][str(config[param_1])] = {}
            if str(config[param_2]) not in full_scores[config["dataset_name"]][str(config[param_1])]:
                if param_3 is None:
                    full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])] = []
                else:
                    full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])] = {}
            if param_3 is not None and str(config[param_3]) not in full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])]:
                full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])][str(config[param_3])] = []
            # print("LEN", len(all_scores))
            if len(all_scores) == 0:
                continue
            last_score_idx = -10
            if "antmaze" in config["dataset_name"]:
                last_score_idx = -5
            if only_last:
                last_score_idx = -1
            # print(len(all_scores), all_scores[last_score_idx:])
            full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])].append(
                np.mean(all_scores[last_score_idx:]))
            config_list.append(config)
            name_list.append(run.name)
            lasts.append(last)

    return full_scores


def get_generalization_data_from_sweeps(sweeps_ids, only_last=True):
    name_list = []
    config_list = []
    full_scores = {}

    for s in tqdm(sweeps_ids, desc="Sweeps processing", position=0, leave=True):
        api = wandb.Api(timeout=39)
        sweep = api.sweep(s)
        runs = sweep.runs
        cur_max = 0
        for run in tqdm(runs, desc="Runs processing", position=0, leave=True):
            all_scores = []
            action_noise_scores = []
            state_noise_scores = []
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            skip = False
            if skip:
                continue
            # print(run.name, end=' ')
            for i, row in run.history(keys=["eval/normalized_score_mean", "eval/normalized_score_mean_sn_0.0_an_0.2", "eval/normalized_score_mean_sn_0.05_an_0.0"], samples=3000).iterrows():
                all_scores.append(row["eval/normalized_score_mean"])
                action_noise_scores.append(row["eval/normalized_score_mean_sn_0.0_an_0.2"])
                state_noise_scores.append(row["eval/normalized_score_mean_sn_0.05_an_0.0"])
            cur_max = max(cur_max, len(all_scores))
            if config["dataset_name"] not in full_scores:
                full_scores[config["dataset_name"]] = {"scores": [], "an_scores": [], "sn_scores": []}
            # print("LEN", len(all_scores))
            if len(all_scores) == 0:
                continue
            last_score_idx = -10
            if "antmaze" in config["dataset_name"]:
                last_score_idx = -5
            if only_last:
                last_score_idx = -1
            # print(len(all_scores), all_scores[last_score_idx:])
            full_scores[config["dataset_name"]]["scores"].append(
                np.mean(all_scores[last_score_idx:]))
            full_scores[config["dataset_name"]]["an_scores"].append(
                np.mean(action_noise_scores[last_score_idx:]))
            full_scores[config["dataset_name"]]["sn_scores"].append(
                np.mean(state_noise_scores[last_score_idx:]))
            config_list.append(config)
            name_list.append(run.name)

    return full_scores


def get_actor_data_from_sweeps(sweeps_ids, only_last=True, params_filter=dict()):
    name_list = []
    config_list = []
    full_scores = {}
    save_keys = [
        "train_metrics/dead_neurons_frac",
        "train_metrics/feature_norms",
        "train_metrics/feature_means",
        "train_metrics/feature_stds",
        "train_metrics/pca_rank",
        "train_metrics/actor_loss",
        "validation_metrics/dead_neurons_frac",
        "validation_metrics/feature_norms",
        "validation_metrics/feature_means",
        "validation_metrics/feature_stds",
        "validation_metrics/pca_rank",
        "validation_metrics/actor_loss",
    ]
    for s in tqdm(sweeps_ids, desc="Sweeps processing", position=0, leave=True):
        api = wandb.Api(timeout=39)
        sweep = api.sweep(s)
        runs = sweep.runs
        cur_max = 0
        for run in tqdm(runs, desc="Runs processing", position=0, leave=True):
            all_scores = {k: [] for k in save_keys}
            action_noise_scores = []
            state_noise_scores = []
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            skip = False
            for fk in params_filter:
                if config[fk] != params_filter[fk]:
                    skip = True
                    break
            if skip:
                continue
            # print(run.name, end=' ')
            for i, row in run.history(keys=save_keys, samples=3000).iterrows():
                for k in save_keys:
                    all_scores[k].append(row[k])
            if config["dataset_name"] not in full_scores:
                full_scores[config["dataset_name"]] = {k: [] for k in save_keys}
            # print("LEN", len(all_scores))
            last_score_idx = -10
            if "antmaze" in config["dataset_name"]:
                last_score_idx = -5
            if only_last:
                last_score_idx = -1
            # print(len(all_scores), all_scores[last_score_idx:])
            for k in save_keys:
                full_scores[config["dataset_name"]][k].append(
                    np.mean(all_scores[k][last_score_idx:]))
            config_list.append(config)
            name_list.append(run.name)

    return full_scores


def get_plasticity_from_sweeps(sweeps_ids, only_last=True, params_filter=dict()):
    name_list = []
    config_list = []
    full_scores = {}
    save_keys = [
        "plasticity/bc_loss",
        "plasticity/start_loss",
    ]
    for s in tqdm(sweeps_ids, desc="Sweeps processing", position=0, leave=True):
        api = wandb.Api(timeout=39)
        sweep = api.sweep(s)
        runs = sweep.runs
        cur_max = 0
        for run in tqdm(runs, desc="Runs processing", position=0, leave=True):
            all_scores = {k: [] for k in save_keys}
            action_noise_scores = []
            state_noise_scores = []
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            skip = False
            for fk in params_filter:
                if config[fk] != params_filter[fk]:
                    skip = True
                    break
            if skip:
                continue
            # print(run.name, end=' ')
            for i, row in run.history(keys=save_keys, samples=3000).iterrows():
                for k in save_keys:
                    all_scores[k].append(row[k])
            if config["dataset_name"] not in full_scores:
                full_scores[config["dataset_name"]] = {k: [] for k in save_keys}
            # print("LEN", len(all_scores))
            last_score_idx = -10
            if "antmaze" in config["dataset_name"]:
                last_score_idx = -5
            if only_last:
                last_score_idx = -1
            # print(len(all_scores), all_scores[last_score_idx:])
            for k in save_keys:
                full_scores[config["dataset_name"]][k].append(
                    np.mean(all_scores[k][last_score_idx:]))
            config_list.append(config)
            name_list.append(run.name)

    return full_scores


def average_seeds(full_scores, is_td3=False, three_params=False):
    S = 0
    full_means = {}
    bests = {}
    for dataset in full_scores:
        ba, bc, bmean, bstd = 0, 0, 0, 0
        for ac in full_scores[dataset]:
            for cc in full_scores[dataset][ac]:
                if not three_params:
                    score = np.mean(full_scores[dataset][ac][cc])
                    std = np.std(full_scores[dataset][ac][cc])
                    if bmean <= score:
                        bmean = score
                        bstd = std
                        ba = ac
                        bc = cc
                    if dataset not in full_means:
                        full_means[dataset] = {}
                    ka = ac
                    if cc not in full_means[dataset]:
                        full_means[dataset][cc] = {}
                    full_means[dataset][cc][ka] = score
                else:
                    for tp in full_scores[dataset][ac][cc]:
                        score = np.mean(full_scores[dataset][ac][cc][tp])
                        std = np.std(full_scores[dataset][ac][cc][tp])
                        if dataset not in full_means:
                            full_means[dataset] = {}
                        ka = ac
                        if cc not in full_means[dataset]:
                            full_means[dataset][cc] = {}
                        if ka not in full_means[dataset][cc]:
                            full_means[dataset][cc][ka] = {}
                        full_means[dataset][cc][ka][tp] = score
        bests[dataset] = {}
        S += bmean
    return full_means


domain2envs = {
    "Gym-MuJoCo": ["hopper", "walker2d", "halfcheetah"],
    "AntMaze": ["antmaze"],
    "Adroit": ["pen", "door", "hammer", "relocate"]
}


def average_domains(full_means, domains_to_proc=["Gym-MuJoCo", "AntMaze", "Adroit"], three_params=False):
    domain_avgereged = {}

    unique_cc = {
        "Gym-MuJoCo": None,
        "AntMaze": None,
        "Adroit": None
    }
    unique_ac = {
        "Gym-MuJoCo": None,
        "AntMaze": None,
        "Adroit": None
    }
    if three_params:
        unique_tp = {
            "Gym-MuJoCo": None,
            "AntMaze": None,
            "Adroit": None
        }

    # print(list(full_means.keys()))
    unique_cc["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"].keys())
    unique_ac["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"][unique_cc["Gym-MuJoCo"][0]].keys())
    if three_params:
        unique_tp["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"][unique_cc["Gym-MuJoCo"][0]][unique_ac["Gym-MuJoCo"][0]].keys())

    unique_cc["AntMaze"] = list(full_means["antmaze-umaze-v2"].keys())
    unique_ac["AntMaze"] = list(full_means["antmaze-umaze-v2"][unique_cc["AntMaze"][0]].keys())
    if three_params:
        unique_tp["AntMaze"] = list(
            full_means["antmaze-umaze-v2"][unique_cc["AntMaze"][0]][unique_ac["AntMaze"][0]].keys())

    unique_cc["Adroit"] = list(full_means["door-expert-v1"].keys())
    unique_ac["Adroit"] = list(full_means["door-expert-v1"][unique_cc["Adroit"][0]].keys())
    if three_params:
        unique_tp["Adroit"] = list(
            full_means["door-expert-v1"][unique_cc["Adroit"][0]][unique_ac["Adroit"][0]].keys())

    for domain in domains_to_proc:
        domain_avgereged[domain] = {}
        for cc in unique_cc[domain]:
            if cc not in domain_avgereged[domain]:
                domain_avgereged[domain][cc] = {}
            for ac in unique_ac[domain]:
                if not three_params:
                    avg = []
                    for data in full_means:
                        is_domain = False
                        for env in domain2envs[domain]:
                            if env in data:
                                is_domain = True
                                break
                        if is_domain:
                            avg.append(full_means[data][cc][ac])
                    domain_avgereged[domain][cc][ac] = np.mean(avg)
                else:
                    if ac not in domain_avgereged[domain][cc]:
                        domain_avgereged[domain][cc][ac] = {}
                    for tp in unique_tp[domain]:
                        avg = []
                        for data in full_means:
                            is_domain = False
                            for env in domain2envs[domain]:
                                if env in data:
                                    is_domain = True
                                    break
                            if is_domain:
                                avg.append(full_means[data][cc][ac][tp])
                        domain_avgereged[domain][cc][ac][tp] = np.mean(avg)

    return domain_avgereged


def listed_avg(data, three_params=False):
    listed_avg = {}

    for env in data:
        if env not in listed_avg:
            listed_avg[env] = []
        for ac in data[env]:
            for cc in data[env][ac]:
                if not three_params:
                    listed_avg[env].append(data[env][ac][cc])
                else:
                    for tp in data[env][ac][cc]:
                        listed_avg[env].append(data[env][ac][cc][tp])
    return listed_avg


def convert_to_lists(full_means, domains_avg, three_params=False):
    listed_all = {}
    listed_domains = {}
    for algo in full_means:
        listed_all[algo] = listed_avg(full_means[algo], three_params=three_params)
        listed_domains[algo] = listed_avg(domains_avg[algo], three_params=three_params)

    return listed_all, listed_domains


def download_data(algo_to_sweeps, param_1="actor_bc_coef", param_2="critic_bc_coef", param_3=None, to_list=True, domains_to_process=["Gym-MuJoCo", "AntMaze", "Adroit"]):
    data = {}
    for algo in algo_to_sweeps:
        print(f"Downloading {algo} data")
        # if "IQL" in algo:
        #     data[algo] = get_data_from_sweeps_iql(algo_to_sweeps[algo])
        # else:    
        data[algo] = get_data_from_sweeps(algo_to_sweeps[algo], param_1, param_2, param_3)

    full_means = {}
    for algo in data:
        full_means[algo] = average_seeds(data[algo], "TD3" in algo, param_3 is not None)

    domains_avg = {}
    for algo in full_means:
        domains_avg[algo] = average_domains(full_means[algo], domains_to_proc=domains_to_process, three_params=param_3 is not None)

    if to_list:
        return convert_to_lists(full_means, domains_avg, three_params=param_3 is not None)
    else:
        return full_means, domains_avg


def transpose_dict(d):
    transposed = {}

    for sub_key in next(iter(d.values())):
        transposed[sub_key] = {}

    for main_key, sub_dict in d.items():
        for sub_key, value in sub_dict.items():
            transposed[sub_key][main_key] = value

    return transposed


def convert_impact_list_to_tex(impact, algo):
    algo = "{" + algo + "}"
    return "& \\textbf{" + algo + "} & " + " & ".join(map(str, impact)) + " \\\\"


def print_tables(data, algorithms, points=[0, 1, 2, 4, 9, 14, 19]):
    # algorithms = ["TD3 + BC", "IQL", "ReBRAC"]
    print(algorithms)
    fst_algo = algorithms[0]

    all_keys = list(sorted(data[fst_algo].keys()))
    all_values = {
        algo: [data[algo][n] if n in data[algo] else data[algo]["Gym-MuJoCo"] for n in all_keys] for algo in algorithms
    }
    for i, name in enumerate(all_keys):
        print("=" * 30)
        print(name)
        print()
        rewards = [data[algo][name] if name in data[algo] else data[algo]["Gym-MuJoCo"] for algo in algorithms]
        max_runs = max(map(len, rewards))
        x = np.arange(max_runs) + 1
        for algo, reward in zip(algorithms, rewards):
            perf = expected_online_performance(reward, len(reward))
            means = np.array(perf['mean'])
            stds = np.array(perf['std'])
            print("& \\textbf{" + algo + "} &", end=" ")
            for point in points:
                if point >= len(reward):
                    print("-", end=(" & " if point != points[-1] else "\\\\\n"))
                else:
                    print("{:3.1f}".format(means[point]), "$\pm$", "{:3.1f}".format(stds[point]),
                          end=(" & " if point != points[-1] else "\\\\\n"))


def print_v_tables(data, algorithms):
    # algorithms = ["TD3 + BC", "IQL", "ReBRAC"]
    print(algorithms)
    fst_algo = algorithms[0]

    locomotion_envs = ["halfcheetah", "hopper", "walker2d"]
    adroit_envs = ["pen", "door", "hammer", "relocate"]

    locomotion_datasets = [
        "random-v2",
        "medium-v2",
        "expert-v2",
        "medium-expert-v2",
        "medium-replay-v2",
        "full-replay-v2",
    ]
    antmaze_datasets = [
        "umaze-v2",
        "medium-play-v2",
        "large-play-v2",
        "umaze-diverse-v2",
        "medium-diverse-v2",
        "large-diverse-v2",
    ]
    adroit_datasets = [
        "human-v1",
        "cloned-v1",
        "expert-v1",
    ]

    concated = {
        env: {} for env in locomotion_envs + ["antmaze"] + adroit_envs
    }

    for env in locomotion_envs:
        for dataset in locomotion_datasets:
            concated[env][dataset] = ["" for _ in range(20)]
    for env in ["antmaze"]:
        for dataset in antmaze_datasets:
            concated[env][dataset] = ["" for _ in range(20)]
    for env in adroit_envs:
        for dataset in adroit_datasets:
            concated[env][dataset] = ["" for _ in range(20)]

    all_keys = list(sorted(data[fst_algo].keys()))
    all_values = {
        algo: [data[algo][n] if n in data[algo] else data[algo]["Gym-MuJoCo"] for n in all_keys] for algo in algorithms
    }
    for i, name in enumerate(all_keys):
        env_name = name.split('-')[0]
        dataset_name = '-'.join(name.split('-')[1:])

        rewards = [data[algo][name] for algo in algorithms]
        max_runs = max(map(len, rewards))
        x = np.arange(max_runs) + 1
        for point in range(20):
            alg_n = 0
            max_idx = 0
            max_val = -10
            strings = []
            for algo, reward in zip(algorithms, rewards):
                perf = expected_online_performance(reward, len(reward))
                means = np.array(perf['mean'])
                stds = np.array(perf['std'])
                if point >= len(means):
                    strings.append("-")
                else:
                    strings.append("{:3.1f}".format(means[point]) + " $\\pm$ " + "{:3.1f}".format(stds[point]))
                    if max_val < means[point]:
                        max_val = means[point]
                        max_idx = alg_n
                alg_n += 1
            strings[max_idx] = "\\textbf{" + strings[max_idx] + "}"
            concated[env_name][dataset_name][point] = " & ".join(strings) + " & "

    for envs, datasets in zip([locomotion_envs, ["antmaze"], adroit_envs],
                              [locomotion_datasets, antmaze_datasets, adroit_datasets]):
        for env in envs:
            print("=" * 30)
            print(env)
            for i in range(20):
                print(i + 1, "&", ("".join([concated[env][dataset][i] for dataset in datasets]))[:-2], "\\\\")


def flatten(data, target_lens=5):
    flat = []
    for env in data:
        env_list = []
        for cc in data[env]:
            for ac in data[env][cc]:
                env_list += data[env][cc][ac]
        while len(env_list) < target_lens:
            env_list.append(np.mean(env_list))
        if len(env_list) > target_lens:
            env_list = env_list[:target_lens]
        flat.append(env_list)
    return flat


def average_scores(data):
    avgs = []
    for task in data:
        for p1 in data[task]:
            for p2 in data[task][p1]:
                avg = np.mean(data[task][p1][p2])
                # print(task, avg)
                avgs.append(avg)
    return np.mean(avgs)



# profiles_data = {}
# profiles_data['ReBRAC'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/0z8dn2l0'], only_last=False)
# profiles_data['ReBRAC+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gs6rtiyt'], only_last=False)
# profiles_data['ReBRAC+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.001})
# profiles_data['ReBRAC+L1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 1e-05})
# profiles_data['ReBRAC+EN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.0001})
# profiles_data['ReBRAC+DO'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/4n10ihzb'], only_last=False, params_filter={"actor_dropout": 0.1})
# profiles_data['ReBRAC+InN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.003})
# profiles_data['ReBRAC+BCN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.01})
# profiles_data['ReBRAC+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.01})
# # profiles_data['ReBRAC+'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/'], only_last=False, params_filter={"actor_wd": })
#
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/profiles_data.pickle', 'rb') as handle:
    profiles_data = pickle.load(handle)

# profiles_data['ReBRAC+DO 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/4n10ihzb'], only_last=False, params_filter={"actor_dropout": 0.1})
# profiles_data['ReBRAC+DO 0.2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/4n10ihzb'], only_last=False, params_filter={"actor_dropout": 0.2})
# profiles_data['ReBRAC+DO 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/4n10ihzb'], only_last=False, params_filter={"actor_dropout": 0.3})
# profiles_data['ReBRAC+DO 0.5'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ozpwee0i'], only_last=False, params_filter={"actor_dropout": 0.5})
# profiles_data['ReBRAC+DO 0.75'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ozpwee0i'], only_last=False, params_filter={"actor_dropout": 0.75})
# profiles_data['ReBRAC+DO 0.9'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ozpwee0i'], only_last=False, params_filter={"actor_dropout": 0.9})

# profiles_data['ReBRAC+FN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/n1zzv2t0'], only_last=False, params_filter={})
# profiles_data['ReBRAC+GN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/0glbugjq'], only_last=False, params_filter={})
# profiles_data['ReBRAC+SN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/1lc3xgkm'], only_last=False, params_filter={})
#
#
# profiles_data['ReBRAC+L2 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.00001})
# profiles_data['ReBRAC+L2 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.0001})
# profiles_data['ReBRAC+L2 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.001})
# profiles_data['ReBRAC+L2 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.01})
# profiles_data['ReBRAC+L2 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=False, params_filter={"actor_wd": 0.1})
#
# profiles_data['ReBRAC+L1 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 0.00001})
# profiles_data['ReBRAC+L1 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 0.0001})
# profiles_data['ReBRAC+L1 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 0.001})
# profiles_data['ReBRAC+L1 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 0.01})
# profiles_data['ReBRAC+L1 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=False, params_filter={"actor_wd": 0.1})
#
# profiles_data['ReBRAC+EN 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.00001})
# profiles_data['ReBRAC+EN 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.0001})
# profiles_data['ReBRAC+EN 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.001})
# profiles_data['ReBRAC+EN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.01})
# profiles_data['ReBRAC+EN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=False, params_filter={"actor_wd": 0.1})
#
#
# profiles_data['ReBRAC+InN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.003})
# profiles_data['ReBRAC+InN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.01})
# profiles_data['ReBRAC+InN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.03})
# profiles_data['ReBRAC+InN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.1})
# profiles_data['ReBRAC+InN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=False, params_filter={"actor_input_noise": 0.3})
#
# profiles_data['ReBRAC+BCN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.003})
# profiles_data['ReBRAC+BCN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.01})
# profiles_data['ReBRAC+BCN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.03})
# profiles_data['ReBRAC+BCN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.1})
# profiles_data['ReBRAC+BCN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=False, params_filter={"actor_bc_noise": 0.3})
#
# profiles_data['ReBRAC+GrN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.003})
# profiles_data['ReBRAC+GrN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.01})
# profiles_data['ReBRAC+GrN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.03})
# profiles_data['ReBRAC+GrN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.1})
# profiles_data['ReBRAC+GrN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=False, params_filter={"actor_grad_noise": 0.3})

#
# profiles_data['ReBRAC+L2+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/7lxu4c8j'], only_last=False)
# profiles_data['ReBRAC+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/asbsw8r1'], only_last=False)
# profiles_data['ReBRAC+DO+LN+BCN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ve3oqy1z'], only_last=False, params_filter={"actor_bc_noise": 0.01})
# profiles_data['ReBRAC+DO+LN+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5fq1q3bs'], only_last=False, params_filter={"actor_grad_noise": 0.003})
# profiles_data['ReBRAC+DO+LN+InN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/mflso9ma'], only_last=False, params_filter={"actor_input_noise": 0.01})
# profiles_data['ReBRAC+DO+LN+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/z1iyvygn'], only_last=False, params_filter={"actor_wd": 0.0001})
# profiles_data['ReBRAC+EN+DO'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/r3cvll25'], only_last=False, params_filter={})
# profiles_data['ReBRAC+L2+DO'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/mys5pr8e'], only_last=False, params_filter={"actor_dropout": 0.1})
# profiles_data['ReBRAC+DO+Sch'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ztdae0v9'], only_last=False, params_filter={"decay_schedule": "linear"})
# profiles_data['ReBRAC+DO+LN+Sch'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/14uf09tt'], only_last=False, params_filter={"decay_schedule": "linear"})
# profiles_data['ReBRAC+DO+LN+BCN+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/py2k3wml'], only_last=False, params_filter={"actor_wd": 1e-05})
# profiles_data['ReBRAC+EN+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/dy9a4o7j'], only_last=False, params_filter={})
# profiles_data['ReBRAC+EN+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/uqktbz3k'], only_last=False, params_filter={})
# profiles_data['ReBRAC+EN+DO+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/gp7kgpfd'], only_last=False, params_filter={"actor_grad_noise": 0.01})
# profiles_data['ReBRAC+EN+DO+BCN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/swmx56bg'], only_last=False, params_filter={"actor_bc_noise": 0.1})
# profiles_data['ReBRAC+DO+LN+GrN+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/fdifae9s'], only_last=False, params_filter={"actor_wd": 0.0001, "l1_ratio": 0.0})
# profiles_data['ReBRAC+DO+LN+GrN+L1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/fdifae9s'], only_last=False, params_filter={"actor_wd": 1e-05, "l1_ratio": 1.0})
# profiles_data['ReBRAC+DO+LN+GrN+EN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/fdifae9s'], only_last=False, params_filter={"actor_wd": 1e-05, "l1_ratio": 0.5})
# profiles_data['ReBRAC+L2+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/bt762wqi'], only_last=False, params_filter={})
# profiles_data['ReBRAC+L1+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/elf58p8y'], only_last=False, params_filter={})
# profiles_data['ReBRAC+L1+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5z21hvjh'], only_last=False, params_filter={})
# profiles_data['ReBRAC+L1+DO'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/x2gvtc4j'], only_last=False, params_filter={})
# profiles_data['ReBRAC+DO+LN+L1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/yros0tt1'], only_last=False, params_filter={"actor_wd": 1e-05, "l1_ratio": 1.0})
# profiles_data['ReBRAC+DO+LN+EN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/yros0tt1'], only_last=False, params_filter={"actor_wd": 1e-05, "l1_ratio": 0.5})


# profiles_data['IQL'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/zb7ph6s6'], only_last=False, param_1="temperature", param_2="expectile")
# profiles_data['IQL+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/6mckizcf'], only_last=False, param_1="temperature", param_2="expectile")
# profiles_data['IQL+FN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/cminzipk'], only_last=False, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qfdwrkqy'], only_last=False, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.1, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+EN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=False, params_filter={"actor_dropout": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+InN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.01}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+BCN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.3}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.01}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/yqsw2jtq'], only_last=False, params_filter={}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO+LN+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/fb40oyxm'], only_last=False, params_filter={"actor_grad_noise": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO+LN+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/3l05qwr6'], only_last=False, params_filter={"actor_wd": 0.1, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO+LN+L1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/3l05qwr6'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO+LN+EN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/3l05qwr6'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
#
# profiles_data['IQL+DO 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=False, params_filter={"actor_dropout": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO 0.2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=False, params_filter={"actor_dropout": 0.2}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=False, params_filter={"actor_dropout": 0.3}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+DO 0.5'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=False, params_filter={"actor_dropout": 0.5}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+L2 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.00001, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L2 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.0001, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L2 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.001, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L2 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L2 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.1, "l1_ratio": 0.0}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+L1 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.00001, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L1 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.0001, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L1 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.001, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L1 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+L1 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.1, "l1_ratio": 1.0}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+EN 0.00001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.00001, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+EN 0.0001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.0001, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+EN 0.001'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.001, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+EN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.01, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+EN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=False, params_filter={"actor_wd": 0.1, "l1_ratio": 0.5}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+InN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.003}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+InN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.01}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+InN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.03}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+InN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+InN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=False, params_filter={"actor_input_noise": 0.3}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+BCN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.003}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+BCN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.01}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+BCN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.03}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+BCN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+BCN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=False, params_filter={"actor_bc_noise": 0.3}, param_1="temperature", param_2="expectile")
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# profiles_data['IQL+GrN 0.003'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.003}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GrN 0.01'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.01}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GrN 0.03'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.03}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GrN 0.1'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.1}, param_1="temperature", param_2="expectile")
# profiles_data['IQL+GrN 0.3'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=False, params_filter={"actor_grad_noise": 0.3}, param_1="temperature", param_2="expectile")

with open('bin/profiles_data.pickle', 'wb') as handle:
    pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/profiles_data.pickle', 'rb') as handle:
    profiles_data = pickle.load(handle)

avg_scores = {k: average_scores(profiles_data[k]) for k in profiles_data}

for k in avg_scores:
    cur_alg = k.split("+")[0]
    print(k, avg_scores[k], (avg_scores[k] - avg_scores[cur_alg]) / avg_scores[cur_alg] * 100)


flat_profiles_data = {k: np.array(flatten(profiles_data[k])) for k in profiles_data}



algorithms = ['ReBRAC', 'ReBRAC+LN', 'ReBRAC+L2', 'ReBRAC+L1', 'ReBRAC+EN', 'ReBRAC+DO', 'ReBRAC+InN', 'ReBRAC+BCN', 'ReBRAC+GrN', ]

normalized_score_dict = {
    'ReBRAC': flat_profiles_data["ReBRAC"].T,
    # 'ReBRAC+LN': (flat_profiles_data["ReBRAC+LN"].T),
    'ReBRAC+L2': (flat_profiles_data["ReBRAC+L2"].T),
    # 'ReBRAC+L1': (flat_profiles_data["ReBRAC+L1"].T),
    # 'ReBRAC+EN': (flat_profiles_data["ReBRAC+EN"].T),
    # 'ReBRAC+DO': (flat_profiles_data["ReBRAC+DO"].T),
    # 'ReBRAC+InN': (flat_profiles_data["ReBRAC+InN"].T),
    # 'ReBRAC+BCN': (flat_profiles_data["ReBRAC+BCN"].T),
    # 'ReBRAC+GrN': (flat_profiles_data["ReBRAC+GrN"].T),
    }

# Human normalized score thresholds
thresholds = np.linspace(-5.0, 150.0, 31)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    normalized_score_dict, thresholds)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
# plt.legend()
plot_utils.plot_performance_profiles(
  score_distributions, thresholds,
  performance_profile_cis=score_distributions_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'D4RL Normalized Score $(\tau)$',
  ax=ax,
  legend=True
  )
plt.savefig("out/perf_profiles.pdf", dpi=300, bbox_inches='tight')
plt.close()

# algorithm_pairs = {
#     'ReBRAC+LN,ReBRAC': (flat_profiles_data["ReBRAC+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L2,ReBRAC': (flat_profiles_data["ReBRAC+L2"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L1,ReBRAC': (flat_profiles_data["ReBRAC+L1"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+EN,ReBRAC': (flat_profiles_data["ReBRAC+EN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO,ReBRAC': (flat_profiles_data["ReBRAC+DO"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+InN,ReBRAC': (flat_profiles_data["ReBRAC+InN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+BCN,ReBRAC': (flat_profiles_data["ReBRAC+BCN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+GrN,ReBRAC': (flat_profiles_data["ReBRAC+GrN"].T, flat_profiles_data["ReBRAC"].T),
#     }
# average_probabilities, average_prob_cis = rly.get_interval_estimates(
#   algorithm_pairs, metrics.probability_of_improvement, reps=2000)
# ax = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
# ax.set_xlim(0.4, 0.9)
# # plt.show()
# plt.savefig("out/improvement_probability.pdf", dpi=300, bbox_inches='tight')


# profiles_data['ReBRAC+L2+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/7lxu4c8j'], only_last=False)
# profiles_data['ReBRAC+DO+LN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/asbsw8r1'], only_last=False)
# profiles_data['ReBRAC+DO+LN+BCN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/ve3oqy1z'], only_last=False, params_filter={"actor_bc_noise": 0.01})
# profiles_data['ReBRAC+DO+LN+GrN'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/5fq1q3bs'], only_last=False, params_filter={"actor_grad_noise": 0.003})
# profiles_data['ReBRAC+DO+LN+L2'] = get_data_from_sweeps(['tarasovd/ActoReg/sweeps/z1iyvygn'], only_last=False, params_filter={"actor_wd": 0.0001})


# algorithm_pairs = {
#     'ReBRAC+L2+LN,ReBRAC': (flat_profiles_data["ReBRAC+L2+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L2+DO,ReBRAC': (flat_profiles_data["ReBRAC+L2+DO"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L2+DO+LN,ReBRAC': (flat_profiles_data["ReBRAC+L2+DO+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L1+LN,ReBRAC': (flat_profiles_data["ReBRAC+L1+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L1+DO,ReBRAC': (flat_profiles_data["ReBRAC+L1+DO"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+L1+DO+LN,ReBRAC': (flat_profiles_data["ReBRAC+L1+DO+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+EN+LN,ReBRAC': (flat_profiles_data["ReBRAC+EN+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+EN+DO,ReBRAC': (flat_profiles_data["ReBRAC+EN+DO"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+EN+DO+LN,ReBRAC': (flat_profiles_data["ReBRAC+EN+DO+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+L2,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+L2"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+BCN,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+BCN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+GrN,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+GrN"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+GrN+L2,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+GrN+L2"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+GrN+L1,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+GrN+L1"].T, flat_profiles_data["ReBRAC"].T),
#     'ReBRAC+DO+LN+GrN+EN,ReBRAC': (flat_profiles_data["ReBRAC+DO+LN+GrN+EN"].T, flat_profiles_data["ReBRAC"].T),
#     }
# average_probabilities, average_prob_cis = rly.get_interval_estimates(
#   algorithm_pairs, metrics.probability_of_improvement, reps=2000)
# ax = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
# ax.set_xlim(0.4, 0.9)
# # plt.show()
# plt.savefig("out/improvement_probability_comb.pdf", dpi=300, bbox_inches='tight')


algorithms = ['ReBRAC', "ReBRAC+L2", 'ReBRAC+DO', 'ReBRAC+L2+LN', 'ReBRAC+DO+LN', 'ReBRAC+DO+LN+BCN', 'ReBRAC+DO+LN+GrN', 'ReBRAC+DO+LN+L2', 'ReBRAC+EN+DO']

normalized_score_dict = {
    'ReBRAC': flat_profiles_data["ReBRAC"].T,
    'ReBRAC+L2': (flat_profiles_data["ReBRAC+L2"].T),
    # 'ReBRAC+DO': (flat_profiles_data["ReBRAC+DO"].T),
    # 'ReBRAC+L2+LN': (flat_profiles_data["ReBRAC+L2+LN"].T),
    # 'ReBRAC+DO+LN': (flat_profiles_data["ReBRAC+DO+LN"].T),
    # 'ReBRAC+DO+LN+BCN': (flat_profiles_data["ReBRAC+DO+LN+BCN"].T),
    'ReBRAC+DO+LN+GrN': (flat_profiles_data["ReBRAC+DO+LN+GrN"].T),
    # 'ReBRAC+DO+LN+L2': (flat_profiles_data["ReBRAC+DO+LN+L2"].T),
    # 'ReBRAC+EN+DO': (flat_profiles_data["ReBRAC+EN+DO"].T),
    }

# Human normalized score thresholds
thresholds = np.linspace(-5.0, 150.0, 31)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    normalized_score_dict, thresholds)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
# plt.legend()
plot_utils.plot_performance_profiles(
  score_distributions, thresholds,
  performance_profile_cis=score_distributions_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'D4RL Normalized Score $(\tau)$',
  ax=ax,
  legend=True
  )
plt.savefig("out/perf_profiles_comb.pdf", dpi=300, bbox_inches='tight')
plt.close()

# algorithms = ['ReBRAC', 'ReBRAC+L2+LN', 'ReBRAC+L2+DO', 'ReBRAC+L2+DO+LN', 'ReBRAC+L1+LN', 'ReBRAC+L1+DO', 'ReBRAC+L1+DO+LN', 'ReBRAC+EN+LN', 'ReBRAC+EN+DO', 'ReBRAC+EN+DO+LN', 'ReBRAC+DO+LN', 'ReBRAC+DO+LN+L2', 'ReBRAC+DO+LN+BCN', 'ReBRAC+DO+LN+GrN', 'ReBRAC+DO+LN+GrN+L2', 'ReBRAC+DO+LN+GrN+L1', 'ReBRAC+DO+LN+GrN+EN', ]#[::-1]
# # Load ALE scores as a dictionary mapping algorithms to their human normalized
# # score matrices, each of which is of size `(num_runs x num_games)`.
# normalized_score_dict = {
#     'ReBRAC': flat_profiles_data["ReBRAC"].T / 100,
#     'ReBRAC+L2+LN': (flat_profiles_data["ReBRAC+L2+LN"].T) / 100,
#     'ReBRAC+L2+DO': (flat_profiles_data["ReBRAC+L2+DO"].T) / 100,
#     'ReBRAC+L2+DO+LN': (flat_profiles_data["ReBRAC+L2+DO+LN"].T) / 100,
#     'ReBRAC+L1+LN': (flat_profiles_data["ReBRAC+L1+LN"].T) / 100,
#     'ReBRAC+L1+DO': (flat_profiles_data["ReBRAC+L1+DO"].T) / 100,
#     'ReBRAC+L1+DO+LN': (flat_profiles_data["ReBRAC+L1+DO+LN"].T) / 100,
#     'ReBRAC+DO+LN': (flat_profiles_data["ReBRAC+DO+LN"].T) / 100,
#     'ReBRAC+DO+LN+L2': (flat_profiles_data["ReBRAC+DO+LN+L2"].T) / 100,
#     'ReBRAC+DO+LN+BCN': (flat_profiles_data["ReBRAC+DO+LN+BCN"].T) / 100,
#     'ReBRAC+DO+LN+GrN': (flat_profiles_data["ReBRAC+DO+LN+GrN"].T) / 100,
#     'ReBRAC+DO+LN+GrN+L2': (flat_profiles_data["ReBRAC+DO+LN+GrN+L2"].T) / 100,
#     'ReBRAC+DO+LN+GrN+L1': (flat_profiles_data["ReBRAC+DO+LN+GrN+L1"].T) / 100,
#     'ReBRAC+DO+LN+GrN+EN': (flat_profiles_data["ReBRAC+DO+LN+GrN+EN"].T) / 100,
#     'ReBRAC+EN+LN': (flat_profiles_data["ReBRAC+EN+LN"].T) / 100,
#     'ReBRAC+EN+DO': (flat_profiles_data["ReBRAC+EN+DO"].T) / 100,
#     'ReBRAC+EN+DO+LN': (flat_profiles_data['ReBRAC+EN+DO+LN'].T) / 100,
#     }
# aggregate_func = lambda x: np.array([
#   metrics.aggregate_median(x),
#   metrics.aggregate_iqm(x),
#   metrics.aggregate_mean(x),
#   metrics.aggregate_optimality_gap(x)])
# aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
#   normalized_score_dict, aggregate_func, reps=50000)
# plot_utils.plot_interval_estimates(
#   aggregate_scores, aggregate_score_cis,
#   metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
#   algorithms=algorithms, xlabel='D4RL Normalized Score / 100')
# plt.savefig("out/metrics_comb.pdf", dpi=300, bbox_inches='tight')
#
#
# algorithms = ['ReBRAC', 'ReBRAC+LN', 'ReBRAC+L2', 'ReBRAC+L1', 'ReBRAC+EN', 'ReBRAC+DO', 'ReBRAC+InN', 'ReBRAC+BCN', 'ReBRAC+GrN', ]
#
# normalized_score_dict = {
#     'ReBRAC': flat_profiles_data["ReBRAC"].T / 100,
#     'ReBRAC+LN': (flat_profiles_data["ReBRAC+LN"].T) / 100,
#     'ReBRAC+L2': (flat_profiles_data["ReBRAC+L2"].T) / 100,
#     'ReBRAC+L1': (flat_profiles_data["ReBRAC+L1"].T) / 100,
#     'ReBRAC+EN': (flat_profiles_data["ReBRAC+EN"].T) / 100,
#     'ReBRAC+DO': (flat_profiles_data["ReBRAC+DO"].T) / 100,
#     'ReBRAC+InN': (flat_profiles_data["ReBRAC+InN"].T) / 100,
#     'ReBRAC+BCN': (flat_profiles_data["ReBRAC+BCN"].T) / 100,
#     'ReBRAC+GrN': (flat_profiles_data["ReBRAC+GrN"].T) / 100,
#     }
# aggregate_func = lambda x: np.array([
#   metrics.aggregate_median(x),
#   metrics.aggregate_iqm(x),
#   metrics.aggregate_mean(x),
#   metrics.aggregate_optimality_gap(x)])
# aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
#   normalized_score_dict, aggregate_func, reps=50000)
# plot_utils.plot_interval_estimates(
#   aggregate_scores, aggregate_score_cis,
#   metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
#   algorithms=algorithms, xlabel='D4RL Normalized Score / 100')
# plt.savefig("out/metrics.pdf", dpi=300, bbox_inches='tight')

def plot_metrics(algorithms, suffix, pi_range=(0.4, 0.9), main_algo="ReBRAC", div_trials=1, profiles=True):
    normalized_score_dict = {
        k: flat_profiles_data[k].T / 100 for k in algorithms
    }
    aggregate_func = lambda x: np.array([
      metrics.aggregate_median(x),
      metrics.aggregate_iqm(x),
      metrics.aggregate_mean(x),
      metrics.aggregate_optimality_gap(x)])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
      normalized_score_dict, aggregate_func, reps=50000 // div_trials)
    plot_utils.plot_interval_estimates(
      aggregate_scores, aggregate_score_cis,
      metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
      algorithms=algorithms, xlabel='D4RL Normalized Score / 100')
    plt.savefig(f"out/metrics_{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    if profiles:
        thresholds = np.linspace(-0.05, 1.5, 31)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            normalized_score_dict, thresholds)
        # Plot score distributions
        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        # plt.legend()
        plot_utils.plot_performance_profiles(
            score_distributions, thresholds,
            performance_profile_cis=score_distributions_cis,
            colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
            xlabel=r'D4RL Normalized Score $(\tau)$',
            ax=ax,
            legend=True
        )
        plt.savefig(f"out/perf_profiles_{suffix}.pdf", dpi=300, bbox_inches='tight')
        plt.close()

    if type(main_algo) is not list:
        algorithms.remove(main_algo)
        algorithm_pairs = {
            f'{k},{main_algo}': (flat_profiles_data[k].T, flat_profiles_data[main_algo].T) for k in algorithms
        }
    else:
        algorithm_pairs = {}
        for ma in main_algo:
            algorithms.remove(ma)
            for alg in algorithms:
                if ma in alg:
                    algorithm_pairs[f'{alg},{ma}'] = (flat_profiles_data[alg].T, flat_profiles_data[ma].T)

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
      algorithm_pairs, metrics.probability_of_improvement, reps=2000 // div_trials)
    ax = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
    ax.set_xlim(pi_range[0], pi_range[1])
    # plt.show()
    plt.savefig(f"out/improvement_probability_{suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

sns.set(style="ticks", font_scale=0.6)
# plot_metrics(
#     ['ReBRAC', 'ReBRAC+L2+LN', 'ReBRAC+L2+DO', 'ReBRAC+L2+DO+LN', 'ReBRAC+L1+LN', 'ReBRAC+L1+DO', 'ReBRAC+L1+DO+LN',
#      'ReBRAC+EN+LN', 'ReBRAC+EN+DO', 'ReBRAC+EN+DO+LN', 'ReBRAC+DO+LN', 'ReBRAC+DO+LN+L2', 'ReBRAC+DO+LN+L1', 'ReBRAC+DO+LN+EN',
#      'ReBRAC+DO+LN+InN', 'ReBRAC+DO+LN+BCN', 'ReBRAC+DO+LN+GrN', 'ReBRAC+DO+LN+GrN+L2', 'ReBRAC+DO+LN+GrN+L1', 'ReBRAC+DO+LN+GrN+EN', ],#[::-1]
#     "comb",
#     profiles=False,
# )
#
# plot_metrics(
#     ['ReBRAC', 'ReBRAC+LN', 'ReBRAC+L2', 'ReBRAC+L1', 'ReBRAC+EN', 'ReBRAC+DO', 'ReBRAC+InN', 'ReBRAC+BCN', 'ReBRAC+GrN', ],#[::-1]
#     "individual",
#     profiles=False,
# )
# plot_metrics(
#     ["ReBRAC", "ReBRAC+DO 0.1", "ReBRAC+DO 0.2", "ReBRAC+DO 0.3", "ReBRAC+DO 0.5", "ReBRAC+DO 0.75", "ReBRAC+DO 0.9"],
#     "dropout",
#     pi_range=(0, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+LN", "ReBRAC+FN", "ReBRAC+GN", "ReBRAC+SN"],
#     "ln",
# )

# plot_metrics(
#     ["ReBRAC", "ReBRAC+L2 0.00001", "ReBRAC+L2 0.0001", "ReBRAC+L2 0.001", "ReBRAC+L2 0.01", "ReBRAC+L2 0.1"],
#     "l2",
#     pi_range=(0.3, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+L1 0.00001", "ReBRAC+L1 0.0001", "ReBRAC+L1 0.001", "ReBRAC+L1 0.01", "ReBRAC+L1 0.1"],
#     "l1",
#     pi_range=(0.3, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+EN 0.00001", "ReBRAC+EN 0.0001", "ReBRAC+EN 0.001", "ReBRAC+EN 0.01", "ReBRAC+EN 0.1"],
#     "en",
#     pi_range=(0.3, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+InN 0.003", "ReBRAC+InN 0.01", "ReBRAC+InN 0.03", "ReBRAC+InN 0.1", "ReBRAC+InN 0.3"],
#     "inn",
#     pi_range=(0.0, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+BCN 0.003", "ReBRAC+BCN 0.01", "ReBRAC+BCN 0.03", "ReBRAC+BCN 0.1", "ReBRAC+BCN 0.3"],
#     "bcn",
#     pi_range=(0.0, 0.9)
# )
#
# plot_metrics(
#     ["ReBRAC", "ReBRAC+GrN 0.003", "ReBRAC+GrN 0.01", "ReBRAC+GrN 0.03", "ReBRAC+GrN 0.1", "ReBRAC+GrN 0.3"],
#     "grn",
#     pi_range=(0.0, 0.9)
# )

# plot_metrics(
#     ['IQL', 'IQL+LN', 'IQL+L2', 'IQL+L1', 'IQL+EN', 'IQL+DO', 'IQL+InN', 'IQL+BCN', 'IQL+GrN'],
#     "iql",
#     pi_range=(0.4, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ['IQL', 'IQL+DO+LN', 'IQL+DO+LN+L2', 'IQL+DO+LN+L1', 'IQL+DO+LN+EN', 'IQL+DO+LN+GrN',],
#     "iql_comb",
#     pi_range=(0.3, 0.9),
#     main_algo="IQL",
# )

# plot_metrics(
#     ["IQL", "IQL+DO 0.1", "IQL+DO 0.2", "IQL+DO 0.3", "IQL+DO 0.5"],
#     "iql_dropout",
#     pi_range=(0.2, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+LN", "IQL+FN", "IQL+GN"],
#     "iql_ln",
#     main_algo="IQL",
#     pi_range=(0.4, 0.7),
# )
#
# plot_metrics(
#     ["IQL", "IQL+L2 0.00001", "IQL+L2 0.0001", "IQL+L2 0.001", "IQL+L2 0.01", "IQL+L2 0.1"],
#     "iql_l2",
#     pi_range=(0.4, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+L1 0.00001", "IQL+L1 0.0001", "IQL+L1 0.001", "IQL+L1 0.01", "IQL+L1 0.1"],
#     "iql_l1",
#     pi_range=(0.1, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+EN 0.00001", "IQL+EN 0.0001", "IQL+EN 0.001", "IQL+EN 0.01", "IQL+EN 0.1"],
#     "iql_en",
#     pi_range=(0.3, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+InN 0.003", "IQL+InN 0.01", "IQL+InN 0.03", "IQL+InN 0.1", "IQL+InN 0.3"],
#     "iql_inn",
#     pi_range=(0.2, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+BCN 0.003", "IQL+BCN 0.01", "IQL+BCN 0.03", "IQL+BCN 0.1", "IQL+BCN 0.3"],
#     "iql_bcn",
#     pi_range=(0.4, 0.8),
#     main_algo="IQL",
# )
#
# plot_metrics(
#     ["IQL", "IQL+GrN 0.003", "IQL+GrN 0.01", "IQL+GrN 0.03", "IQL+GrN 0.1", "IQL+GrN 0.3"],
#     "iql_grn",
#     pi_range=(0.4, 0.8),
#     main_algo="IQL",
# )


# dataset_scale_data = {}

# dataset_scale_data['ReBRAC'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/njrnd5dc'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
# print(dataset_scale_data)
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('bin/scale_data.pickle', 'rb') as handle:
#     dataset_scale_data = pickle.load(handle)
#
# dataset_scale_data['ReBRAC+DO'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/6eu5206d'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
#
#
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# dataset_scale_data['ReBRAC+LN'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/3ol0epj4'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
#
#
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# dataset_scale_data['ReBRAC+L2'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/j3k3pq9u'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
#
#
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# dataset_scale_data['ReBRAC+L1'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/tx0mddo3'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
#
#
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# dataset_scale_data['ReBRAC+EN'] = [
#     average_scores(get_data_from_sweeps(
#         ['tarasovd/ActoReg/sweeps/4pkra3ma'], only_last=False, params_filter={"validation_frac": vf})
#     )
#     for vf in [0.1, 0.25, 0.5, 0.75, 0.9][::-1]
# ]
# with open('bin/scale_data.pickle', 'wb') as handle:
#     pickle.dump(dataset_scale_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('bin/scale_data.pickle', 'rb') as handle:
#     dataset_scale_data = pickle.load(handle)
#
# for k in dataset_scale_data:
#     plt.plot([0.1, 0.25, 0.5, 0.75, 0.9], dataset_scale_data[k], label=k)
# plt.grid()
# plt.legend()
# plt.savefig(f"out/data_scale_rebrac.pdf", dpi=300, bbox_inches='tight')
# plt.close()

# generalization_data = {}
# generalization_data["ReBRAC"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/36w1lih9', 'tarasovd/ActoReg/wy58h3gg'], only_last=True)
# generalization_data["ReBRAC+L2"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/j16hbmaq', 'tarasovd/ActoReg/i9xlh0ch'], only_last=True)
# generalization_data["ReBRAC+DO+LN+GrN"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/j2qnvwil', 'tarasovd/ActoReg/nawvkbeq'], only_last=True)
#
# with open('bin/gener_data.pickle', 'wb') as handle:
#     pickle.dump(generalization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/gener_data.pickle', 'rb') as handle:
#     generalization_data = pickle.load(handle)

# generalization_data["IQL"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/1h2z4kzd', 'tarasovd/ActoReg/2c8gjdhb'], only_last=True)
# generalization_data["IQL+L1"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/08ygop65', 'tarasovd/ActoReg/rrtuyx4a'], only_last=True)
# generalization_data["IQL+DO+LN+EN"] = get_generalization_data_from_sweeps(['tarasovd/ActoReg/sweeps/oadnmq99', 'tarasovd/ActoReg/01ae51st'], only_last=True)

# with open('bin/gener_data.pickle', 'wb') as handle:
#     pickle.dump(generalization_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/gener_data.pickle', 'rb') as handle:
    generalization_data = pickle.load(handle)

ordered_envs = [
    "hopper-random-v2",
    "hopper-medium-v2",
    "hopper-expert-v2",
    "hopper-medium-expert-v2",
    "hopper-medium-replay-v2",
    "hopper-full-replay-v2",
    "halfcheetah-random-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-expert-v2",
    "halfcheetah-medium-expert-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-full-replay-v2",
    "walker2d-random-v2",
    "walker2d-medium-v2",
    "walker2d-expert-v2",
    "walker2d-medium-expert-v2",
    "walker2d-medium-replay-v2",
    "walker2d-full-replay-v2",
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
]

def make_table(scores):
    print("=" * 80)
    algos = list(scores.keys())
    envs = ordered_envs
    print(" & ".join(algos) + " \\\\")
    gym_scores = [[] for _ in range(len(algos))]
    antmaze_scores = [[] for _ in range(len(algos))]
    adroit_scores = [[] for _ in range(len(algos))]
    adroit_ne_scores = [[] for _ in range(len(algos))]
    all_scores = [[] for _ in range(len(algos))]
    for env in envs:
        print(env, end=" & ")
        for i, a in enumerate(algos):
            print(f"{np.mean(scores[a][env]['scores']):.1f} $\\pm$ {np.std(scores[a][env]['scores']):.1f}", end=" & ")
            all_scores[i].append(np.mean(scores[a][env]['scores']))
            if "antmaze" in env:
                antmaze_scores[i].append(np.mean(scores[a][env]['scores']))
            elif "v2" in env:
                gym_scores[i].append(np.mean(scores[a][env]['scores']))
            else:
                if "expert" not in env:
                    adroit_ne_scores[i].append(np.mean(scores[a][env]['scores']))
                adroit_scores[i].append(np.mean(scores[a][env]['scores']))
        print("\\\\")
    for domain, avgs in zip(["Gym-MuJoCo", "AntMaze", "Adroit w\\o expert", "Adroit", "Avg"], [gym_scores, antmaze_scores, adroit_ne_scores, adroit_scores, all_scores]):
        print(domain, end=" & ")
        for i, a in enumerate(algos):
            print(f"{np.mean(avgs[i]):.1f} $\\pm$ {np.std(avgs[i]):.1f}", end=" & ")
        print("\\\\")
    print("=" * 80)

make_table(generalization_data)


def flatten_pd(data, target_lens=5):
    flat = []
    for env in ordered_envs:
        env_list = []
        env_list += data[env]['scores']
        while len(env_list) < target_lens:
            env_list.append(np.mean(env_list))
        if len(env_list) > target_lens:
            env_list = env_list[:target_lens]
        flat.append(env_list)
    return flat

# sns.set(style="ticks", font_scale=0.7)
#
# flat_perdataset = {k: np.array(flatten_pd(generalization_data[k])) for k in generalization_data}
# plot_metrics(
#     ["ReBRAC", "ReBRAC+L2", "ReBRAC+DO+LN+GrN", "IQL", "IQL+L1", "IQL+DO+LN+EN"],
#     "per_dataset",
#     pi_range=(0.4, 0.9),
#     main_algo=["ReBRAC", "IQL"],
#     # div_trials=100
# )
def make_table_generalization(scores, noise_field="an_scores"):
    print("=" * 80)
    algos = list(scores.keys())
    envs = ordered_envs
    print(" & ".join(algos) + " \\\\")
    gym_scores = [[] for _ in range(len(algos))]
    antmaze_scores = [[] for _ in range(len(algos))]
    adroit_scores = [[] for _ in range(len(algos))]
    adroit_ne_scores = [[] for _ in range(len(algos))]
    all_scores = [[] for _ in range(len(algos))]

    gym_unag = [[] for _ in range(len(algos))]
    antmaze_unag = [[] for _ in range(len(algos))]
    adroit_unag = [[] for _ in range(len(algos))]
    all_unag = [[] for _ in range(len(algos))]

    for env in envs:
        print(env, end=" & ")
        for i, a in enumerate(algos):
            fraction = np.minimum(np.maximum(scores[a][env][noise_field], 1e-6) / np.maximum(scores[a][env]['scores'], 1e-6), 1.1)
            # print(fraction, np.mean(fraction))
            print(f"{np.mean(fraction):.3f} \\pm {np.std(fraction):.3f}", end=" & ")
            all_scores[i].append(np.mean(fraction))
            all_unag[i] += list(fraction)
            if "antmaze" in env:
                antmaze_scores[i].append(np.mean(fraction))
                antmaze_unag[i] += list(fraction)
            elif "v2" in env:
                gym_scores[i].append(np.mean(fraction))
                gym_unag[i] += list(fraction)
            else:
                if "expert" not in env:
                    adroit_ne_scores[i].append(fraction)
                adroit_scores[i].append(np.mean(fraction))
                adroit_unag[i] += list(fraction)
        print("\\\\")
    for domain, avgs in zip(["Gym-MuJoCo", "AntMaze", "Adroit w\\o expert", "Adroit", "Avg"], [gym_scores, antmaze_scores, adroit_ne_scores, adroit_scores, all_scores]):
        print(domain, end=" & ")
        for i, a in enumerate(algos):
            print(f"{np.mean(avgs[i]):.3f} \\pm {np.std(avgs[i]):.3f}", end=" & ")
        print("\\\\")
    print("=" * 80)
    return algos, gym_unag, antmaze_unag, adroit_unag, all_unag


labels, gym_scores_a, antmaze_scores_a, adroit_scores_a, all_scores_a = make_table_generalization(generalization_data, "an_scores")
labels, gym_scores_s, antmaze_scores_s, adroit_scores_s, all_scores_s = make_table_generalization(generalization_data, "sn_scores")

colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'pink', 'tan']

def plot_boxes(data, labels, title, save_suffix):
    box = plt.boxplot(data, labels=labels, patch_artist=True, vert=False, showfliers=False)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')
    # Add a title and labels
    plt.title(title)
    plt.xlabel("Fraction of original performance")
    # Display the plot
    plt.savefig(f"out/boxes_{save_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


# plot_boxes(gym_scores_a, labels, "Gym-MuJoCo + actions noise", "gym_an")
# plot_boxes(antmaze_scores_a, labels, "AntMaze + actions noise", "antmaze_an")
# plot_boxes(adroit_scores_a, labels, "Adroit + actions noise", "adroit_an")
# plot_boxes(all_scores_a, labels, "All datasets + actions noise", "all_an")
# plot_boxes(gym_scores_s, labels, "Gym-MuJoCo + states noise", "gym_sn")
# plot_boxes(antmaze_scores_s, labels, "AntMaze + states noise", "antmaze_sn")
# plot_boxes(adroit_scores_s, labels, "Adroit + states noise", "adroit_sn")
# plot_boxes(all_scores_s, labels, "All datasets + states noise", "all_sn")


# actor_data = {}
# actor_data['ReBRAC'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/0z8dn2l0'], only_last=True)
# actor_data['ReBRAC+LN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/gs6rtiyt'], only_last=True)
# actor_data['ReBRAC+L2'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/l367pg9o'], only_last=True, params_filter={"actor_wd": 0.001})
# actor_data['ReBRAC+L1'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/psuls6h7'], only_last=True, params_filter={"actor_wd": 0.0001})
# actor_data['ReBRAC+EN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/h6pbjmbh'], only_last=True, params_filter={"actor_wd": 0.001})
# actor_data['ReBRAC+DO'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/ozpwee0i'], only_last=True, params_filter={"actor_dropout": 0.1})
# actor_data['ReBRAC+InN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/vxdbh1ir'], only_last=True, params_filter={"actor_input_noise": 0.003})
# actor_data['ReBRAC+BCN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/gaf8jov8'], only_last=True, params_filter={"actor_bc_noise":  0.01})
# actor_data['ReBRAC+GrN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/40y41cde'], only_last=True, params_filter={"actor_grad_noise": 0.01})
#
# actor_data['IQL'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/zb7ph6s6'], only_last=True)
# actor_data['IQL+LN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/6mckizcf'], only_last=True)
# actor_data['IQL+L2'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=True, params_filter={"actor_wd": 0.1, "l1_ratio": 0.0})
# actor_data['IQL+L1'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=True, params_filter={"actor_wd": 0.01, "l1_ratio": 1.0})
# actor_data['IQL+EN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/qqr9mkfd'], only_last=True, params_filter={"actor_wd": 0.01, "l1_ratio": 0.5})
# actor_data['IQL+DO'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/5h7vroey'], only_last=True, params_filter={"actor_dropout": 0.1})
# actor_data['IQL+InN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/k4wksvwn'], only_last=True, params_filter={"actor_input_noise": 0.01})
# actor_data['IQL+BCN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/xha44dyd'], only_last=True, params_filter={"actor_bc_noise": 0.3})
# actor_data['IQL+GrN'] = get_actor_data_from_sweeps(['tarasovd/ActoReg/sweeps/8qeewggq'], only_last=True, params_filter={"actor_grad_noise": 0.01})
#
# with open('bin/actor_data.pickle', 'wb') as handle:
#     pickle.dump(actor_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/actor_data.pickle', 'rb') as handle:
    actor_data = pickle.load(handle)


def preprocess_actor_data(scores, metric="pca_rank"):
    algos = list(scores.keys())
    envs = [
        "hopper-medium-v2",
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-large-diverse-v2",
        "pen-cloned-v1",
        "door-cloned-v1",
        "hammer-cloned-v1",
        "relocate-cloned-v1",
    ]
    gym_scores = [[] for _ in range(len(algos))]
    antmaze_scores = [[] for _ in range(len(algos))]
    adroit_scores = [[] for _ in range(len(algos))]
    adroit_ne_scores = [[] for _ in range(len(algos))]
    all_scores = [[] for _ in range(len(algos))]

    gym_unag = [[] for _ in range(len(algos))]
    antmaze_unag = [[] for _ in range(len(algos))]
    adroit_unag = [[] for _ in range(len(algos))]
    all_unag = [[] for _ in range(len(algos))]
    for env in envs:
        for i, a in enumerate(algos):
            # print(a, env, scores[a][env][f'validation_metrics/{metric}'])
            fraction = (np.array(scores[a][env][f'validation_metrics/{metric}'])) / np.maximum(scores[a][env][f'train_metrics/{metric}'], 1e-3)
            lf = list(filter(lambda x: x < 10, list(fraction)))
            # print(fraction, np.mean(fraction))
            all_unag[i] += lf
            if "antmaze" in env:
                antmaze_scores[i].append(np.mean(fraction))
                antmaze_unag[i] += lf
            elif "v2" in env:
                gym_scores[i].append(np.mean(fraction))
                gym_unag[i] += lf
            else:
                if "expert" not in env:
                    adroit_ne_scores[i].append(fraction)
                adroit_scores[i].append(np.mean(fraction))
                adroit_unag[i] += lf
    return algos, gym_unag, antmaze_unag, adroit_unag, all_unag


def plot_actor_boxes(data, labels, title, save_suffix, xlabel="validation / train"):
    box = plt.boxplot(data, labels=labels, patch_artist=True, vert=False, showfliers=False)
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('red')
    # Add a title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    # Display the plot
    plt.savefig(f"out/actor_boxes_{save_suffix}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric(actor_data, metric, title, suffix):
    labels, gym_scores, antmaze_scores, adroit_scores, all_scores = preprocess_actor_data(actor_data, metric)

    for sc, t, s in zip(
        [gym_scores, antmaze_scores, adroit_scores, all_scores],
        ["Gym-MuJoCo ", "AntMaze ", "Adroit ", "All domains "],
        ["gym", "antmaze", "adroit", "all"]
    ):
        plot_actor_boxes(sc, labels, t + title, f"{suffix}_{s}")


sns.set(style="ticks", font_scale=1.0)
plot_metric(actor_data, "dead_neurons_frac", "Dead neurons", "dead_neurons")
plot_metric(actor_data, "feature_norms", "Features norms", "feature_norms")
plot_metric(actor_data, "feature_means", "Features means", "feature_means")
plot_metric(actor_data, "feature_stds", "Features stds", "feature_stds")
plot_metric(actor_data, "pca_rank", "PCA rank", "pca")
plot_metric(actor_data, "actor_loss", "Actor loss", "loss")

# plasticity_data = {}
# plasticity_data["ReBRAC"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/hhrqhme8'], only_last=True)
# plasticity_data["ReBRAC+LN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/8k5oxdy2'], only_last=True)
# plasticity_data["ReBRAC+L2"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/7cvugreb'], only_last=True)
# plasticity_data["ReBRAC+L1"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/r7qq8yx1'], only_last=True)
# plasticity_data["ReBRAC+EN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/nq4h6mvl'], only_last=True)
# plasticity_data["ReBRAC+DO"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/21vk0egg'], only_last=True)
# plasticity_data["ReBRAC+InN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/5n3zz22m'], only_last=True)
# plasticity_data["ReBRAC+BCN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/jnasq4rs'], only_last=True)
# plasticity_data["ReBRAC+GrN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/2rkoojnp'], only_last=True)
#
# with open('bin/plasticity_data.pickle', 'wb') as handle:
#     pickle.dump(plasticity_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/plasticity_data.pickle', 'rb') as handle:
    plasticity_data = pickle.load(handle)

plasticity_data["IQL"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/sfck1iif'], only_last=True)
plasticity_data["IQL+LN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/4dbfwbfh'], only_last=True)
plasticity_data["IQL+L2"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/nxl9f4i4'], only_last=True)
plasticity_data["IQL+L1"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/ug8ajlq6'], only_last=True)
plasticity_data["IQL+EN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/p1k9lmnw'], only_last=True)
plasticity_data["IQL+DO"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/3nx833sc'], only_last=True)
plasticity_data["IQL+InN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/y599n4z4'], only_last=True)
plasticity_data["IQL+BCN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/ust6a7fh'], only_last=True)
plasticity_data["IQL+GrN"] = get_plasticity_from_sweeps(['tarasovd/ActoReg/sweeps/eepfpvya'], only_last=True)

with open('bin/plasticity_data.pickle', 'wb') as handle:
    pickle.dump(plasticity_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/plasticity_data.pickle', 'rb') as handle:
    plasticity_data = pickle.load(handle)

def preprocess_plasticity_data(scores, metric="bc_loss"):
    algos = list(scores.keys())
    envs = [
        "hopper-medium-v2",
        "halfcheetah-medium-v2",
        "walker2d-medium-v2",
        "antmaze-medium-diverse-v2",
        "antmaze-large-diverse-v2",
        "pen-cloned-v1",
        "door-cloned-v1",
        "hammer-cloned-v1",
        "relocate-cloned-v1",
    ]
    gym_scores = [[] for _ in range(len(algos))]
    antmaze_scores = [[] for _ in range(len(algos))]
    adroit_scores = [[] for _ in range(len(algos))]
    adroit_ne_scores = [[] for _ in range(len(algos))]
    all_scores = [[] for _ in range(len(algos))]

    gym_unag = [[] for _ in range(len(algos))]
    antmaze_unag = [[] for _ in range(len(algos))]
    adroit_unag = [[] for _ in range(len(algos))]
    all_unag = [[] for _ in range(len(algos))]
    for env in envs:
        for i, a in enumerate(algos):
            # print(a, env, scores[a][env][f'validation_metrics/{metric}'])
            fraction = np.array(scores[a][env][f'plasticity/{metric}'])
            lf = list(filter(lambda x: True, list(fraction)))
            # print(fraction, np.mean(fraction))
            all_unag[i] += lf
            if "antmaze" in env:
                antmaze_scores[i].append(np.mean(fraction))
                antmaze_unag[i] += lf
            elif "v2" in env:
                gym_scores[i].append(np.mean(fraction))
                gym_unag[i] += lf
            else:
                if "expert" not in env:
                    adroit_ne_scores[i].append(fraction)
                adroit_scores[i].append(np.mean(fraction))
                adroit_unag[i] += lf
    return algos, gym_unag, antmaze_unag, adroit_unag, all_unag

labels, gym_scores, antmaze_scores, adroit_scores, all_scores = preprocess_plasticity_data(plasticity_data)
for sc, t, s in zip(
    [gym_scores, antmaze_scores, adroit_scores, all_scores],
    ["Gym-MuJoCo ", "AntMaze ", "Adroit ", "All domains "],
    ["gym", "antmaze", "adroit", "all"]
):
    plot_actor_boxes(sc, labels, t + "Plasticity", f"plasticity_{s}", xlabel="Optimized BC loss")

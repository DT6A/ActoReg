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
#
# with open('bin/profiles_data.pickle', 'wb') as handle:
#     pickle.dump(profiles_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('bin/profiles_data.pickle', 'rb') as handle:
#     profiles_data = pickle.load(handle)

avg_scores = {k: average_scores(profiles_data[k]) for k in profiles_data}

for k in avg_scores:
    print(k, avg_scores[k], (avg_scores[k] - avg_scores["ReBRAC"]) / avg_scores["ReBRAC"] * 100)


flat_profiles_data = {k: np.array(flatten(profiles_data[k])) for k in profiles_data}

sns.set(style="ticks", font_scale=0.5)

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

def plot_metrics(algorithms, suffix, pi_range=(0.4, 0.9)):
    normalized_score_dict = {
        k: flat_profiles_data[k].T / 100 for k in algorithms
    }
    aggregate_func = lambda x: np.array([
      metrics.aggregate_median(x),
      metrics.aggregate_iqm(x),
      metrics.aggregate_mean(x),
      metrics.aggregate_optimality_gap(x)])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
      normalized_score_dict, aggregate_func, reps=50000)
    plot_utils.plot_interval_estimates(
      aggregate_scores, aggregate_score_cis,
      metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
      algorithms=algorithms, xlabel='D4RL Normalized Score / 100')
    plt.savefig(f"out/metrics_{suffix}.pdf", dpi=300, bbox_inches='tight')

    algorithms.remove("ReBRAC")
    algorithm_pairs = {
        f'{k},ReBRAC': (flat_profiles_data[k].T, flat_profiles_data["ReBRAC"].T) for k in algorithms
    }
    average_probabilities, average_prob_cis = rly.get_interval_estimates(
      algorithm_pairs, metrics.probability_of_improvement, reps=2000)
    ax = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
    ax.set_xlim(pi_range[0], pi_range[1])
    # plt.show()
    plt.savefig(f"out/improvement_probability_{suffix}.pdf", dpi=300, bbox_inches='tight')


plot_metrics(
    ['ReBRAC', 'ReBRAC+L2+LN', 'ReBRAC+L2+DO', 'ReBRAC+L2+DO+LN', 'ReBRAC+L1+LN', 'ReBRAC+L1+DO', 'ReBRAC+L1+DO+LN',
     'ReBRAC+EN+LN', 'ReBRAC+EN+DO', 'ReBRAC+EN+DO+LN', 'ReBRAC+DO+LN', 'ReBRAC+DO+LN+L2', 'ReBRAC+DO+LN+L1', 'ReBRAC+DO+LN+EN', 'ReBRAC+DO+LN+BCN', 'ReBRAC+DO+LN+GrN',
     'ReBRAC+DO+LN+GrN+L2', 'ReBRAC+DO+LN+GrN+L1', 'ReBRAC+DO+LN+GrN+EN', ],#[::-1]
    "comb",
)

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

# TODO: (DO + Input Noise), (Schedulers?)

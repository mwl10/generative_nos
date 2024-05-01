import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import math
import inspect
import os
from matplotlib.ticker import ScalarFormatter
import numpy as np
import json
import sys
sys.path.append((os.path.dirname(os.path.dirname(__file__))))


class fstr:
    def __init__(self, payload):
        self.payload = payload
    def __str__(self):
        vars = inspect.currentframe().f_back.f_globals.copy()
        vars.update(inspect.currentframe().f_back.f_locals)
        return self.payload.format(**vars)

def scientific_notation(f, num_digits=3):
    return "{:.{}e}".format(f, num_digits)


def l2_vs_size():
    train_tests = ["test", "train"]

    for train_test in train_tests:
        print("="*30)
        save_folder = f"{global_save_folder}/"
        fig, ax = plt.subplots()

        min_y, max_y = 100., -100.

        arr = []
        for n in ns:
            with open(str(vanilla_results_file), "r") as f:
                temp_f = min(json.load(f)[f"{train_test}_l2_error"])
                arr.append(temp_f)

        min_y = min(min_y, min(arr))
        max_y = max(max_y, max(arr))
        ax.plot(ns, arr, linewidth=2.0, marker="v", markersize=15, color="red", label=str(vanilla_label))

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$n$", fontweight="semibold", fontsize=32)
        plt.ylabel(r"$L_2$ relative error", fontweight="semibold", fontsize=32)
        plt.title(f"{train_test.capitalize()}" + r" $L_2$ relative error", fontweight="semibold", fontsize=32)
        plt.xticks(ns, [str(s) for s in ns])
        ax.set_yscale("log")
        plt.yticks([min_y] + [10**i for i in range(int(np.floor(np.log10(min_y))), int(np.ceil(np.log10(max_y)) + 1)) if 10**i != min_y])
        plt.legend(frameon=False, loc="upper right", prop=legend_param, bbox_to_anchor=anchor_tuple)
        fig.savefig(save_folder+f"{train_test}_l2_vs_size.png", bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()

def linf_vs_size():
    train_tests = ["test", "train"]

    for train_test in train_tests:
        print("="*30)
        save_folder = f"{global_save_folder}/"
        fig, ax = plt.subplots()

        min_y, max_y = 100., -100.

        arr = []
        for n in ns:
            with open(str(vanilla_results_file), "r") as f:
                temp_f = min(json.load(f)[f"{train_test}_linf_error"])
                arr.append(temp_f)
        min_y = min(min_y, min(arr))
        max_y = max(max_y, max(arr))
        ax.plot(ns, arr, linewidth=2.0, marker="v", markersize=15, color="red", label=str(vanilla_label))

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        plt.xlabel(r"$n$", fontweight="semibold", fontsize=32)
        plt.ylabel(r"$L_\infty$ relative error", fontweight="semibold", fontsize=32)
        plt.title(f"{train_test.capitalize()}" + r" $L_\infty$ relative error", fontweight="semibold", fontsize=32)
        plt.xticks(ns, [str(s) for s in ns])
        ax.set_yscale("log")
        plt.yticks([min_y] + [10**i for i in range(int(np.floor(np.log10(min_y))), int(np.ceil(np.log10(max_y)) + 1)) if 10**i != min_y])
        plt.legend(frameon=False, loc="upper right", prop=legend_param, bbox_to_anchor=anchor_tuple)
        fig.savefig(save_folder+f"{train_test}_linf_vs_size.png", bbox_inches="tight")
        plt.cla()
        plt.clf()
        plt.close()

def train_loss_vs_size():
    train_test = "training"

    save_folder = f"{global_save_folder}/"
    fig, ax = plt.subplots()

    min_y, max_y = 100., -100.

    arr = []
    for n in ns:
        with open(str(vanilla_results_file), "r") as f:
            temp_f = min(json.load(f)[f"{train_test}_loss"])
            arr.append(temp_f)
    min_y = min(min_y, min(arr))
    max_y = max(max_y, max(arr))

    ax.plot(ns, arr, linewidth=2.0, marker="v", markersize=15, color="red", label=str(vanilla_label))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    plt.xlabel(r"$n$", fontweight="semibold", fontsize=32)
    plt.ylabel(r"Training loss", fontweight="semibold", fontsize=32)
    plt.title(f"Training loss", fontweight="semibold", fontsize=32)
    plt.xticks(ns, [str(s) for s in ns])
    ax.set_yscale("log")
    plt.yticks([min_y] + [10**i for i in range(int(np.floor(np.log10(min_y))), int(np.ceil(np.log10(max_y)) + 1)) if 10**i != min_y])
    plt.legend(frameon=False, loc="upper right", prop=legend_param, bbox_to_anchor=anchor_tuple)
    fig.savefig(save_folder+f"train_loss_vs_size.png", bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()

def epochs_vs_size():
    train_test = "training"

    save_folder = f"{global_save_folder}/"
    fig, ax = plt.subplots()

    min_y, max_y = 100., -100.

    arr = []
    for n in ns:
        with open(str(vanilla_results_file), "r") as f:
            data = json.load(f)
            idx = min(range(len(data[f"{train_test}_loss"])), key=data[f"{train_test}_loss"].__getitem__)
            arr.append(idx)
    min_y = min(min_y, min(arr))
    max_y = max(max_y, max(arr))

    ax.plot(ns, arr, linewidth=2.0, marker="v", markersize=15, color="red", label=str(vanilla_label))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    plt.xlabel(r"$n$", fontweight="semibold", fontsize=32)
    plt.ylabel(r"Epochs", fontweight="semibold", fontsize=32)
    plt.title(f"Training epochs", fontweight="semibold", fontsize=32)
    plt.xticks(ns, [str(s) for s in ns])
    plt.legend(frameon=False, loc="upper right", prop=legend_param, bbox_to_anchor=anchor_tuple)
    fig.savefig(save_folder+f"epochs_vs_size.png", bbox_inches="tight")
    plt.cla()
    plt.clf()
    plt.close()

if __name__ == "__main__":
    plt.rcParams["font.weight"] = "semibold"
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "STIXGeneral:italic"
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.rcParams["ytick.labelsize"] = 23
    plt.rcParams["xtick.labelsize"] = 23
    legend_param = {"weight": "normal"}

    bar_width = 0.01

    markers = [
        'o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x',
        '|', '_', '.', ',', '1', '2', '3', '4', '8', 'h', 'H', 'd', 'D',
        'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ]

    colors = [
        'b', 'g', 'peru', 'c', 'm', 'tab:olive', 'lightgray',
        'tab:blue', 'mediumorchid', 'chocolate', 'pink', 'tab:purple', 'tab:brown',
        'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'xkcd:lavender', 'xkcd:teal',
        'xkcd:coral', 'xkcd:azure', 'xkcd:ruby', 'xkcd:salmon', 'xkcd:plum', 'xkcd:turquoise',
        'xkcd:gold', 'xkcd:lime', 'xkcd:indigo', 'xkcd:peach', 'xkcd:rose', 'xkcd:tan',
        'xkcd:slate', 'xkcd:olive', 'xkcd:mauve', 'xkcd:aqua', 'xkcd:sepia', 'xkcd:steel',
        'xkcd:mustard'
    ]

    anchor_tuple = (1.144, 1)

    # whether to plot or print
    plot_print = "print"

    # top folder depending on which experiments being plotted
    problem_folders = [
        "Darcy_triangular",
        "Advection",
        "Burgers",
    ]

    seed = 0
    ns = [8, 16, 32]

    all_run_folder = "final_results"

    vanilla_label = fstr("n={n}")

    vanilla_results_file = fstr("../vano_results/{problem_folder}/vano_n={n}_seed={seed}.json")

    plot_save_folder = "all_plots"

    for problem_idx, problem_folder in enumerate(problem_folders):
        global_save_folder = f"../{plot_save_folder}/{problem_folder}"

        print("\n" + "="*20)
        print(problem_folder)
        print("\n" + "="*20)
        print("\nl2_vs_size")
        l2_vs_size()

        print("\n" + "="*20)
        print("\nlinf_vs_size")
        linf_vs_size()

        print("\n" + "="*20)
        print("\ntrain_loss_vs_size")
        train_loss_vs_size()

        print("\n" + "="*20)
        print("\nepochs_vs_size")
        epochs_vs_size()


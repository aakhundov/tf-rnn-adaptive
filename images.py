import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


mpl.rcParams['axes.linewidth'] = 0.0
mpl.rcParams['lines.dotted_pattern'] = [1, 1]
mpl.rcParams['lines.scale_dashes'] = False

dpi = 150

epochs = 250
steps_per_epoch = 1000

log_dir = "results/logs/"
eval_dir = "results/evaluation/"
image_dir = "results/images/"

experiments = ["parity", "logic", "addition", "sort"]
time_penalties = ["0.0001", "0.001", "0.01", "0.1", "x"]
penalty_colors = ["#7200FF", "#19E0DC", "#CDE480", "#FF1600", "#000000"]
seeds = {
    "parity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "logic": [0, 3, 6, 8],
    "addition": [0],
    "sort": [0]
}

error_image_y_limits = [0.5, 0.7, 1.0, 0.8]
error_vs_diff_image_y_limits = [0.9, 0.5, 1.0, 0.4]
ponder_vs_diff_image_y_limits = [10, 10, 10, 10]
last_error_image_y_limits = [0.50, 0.30, 0.50, 0.20]
difficulties = [[1, 16], [1, 10], [1, 5], [2, 10]]


errors = {e: np.zeros([
    len(time_penalties), len(seeds[e]), epochs
]) for e in experiments}

error_vs_diff = {e: np.zeros([
    len(time_penalties),
    len(seeds[e]),
    difficulties[i][1] - difficulties[i][0] + 1
]) for i, e in enumerate(experiments)}

ponder_vs_diff = {e: np.zeros([
    len(time_penalties)-1,
    len(seeds[e]),
    difficulties[i][1] - difficulties[i][0] + 1
]) for i, e in enumerate(experiments)}


for e in experiments:
    for p in range(len(time_penalties)):
        for s in range(len(seeds[e])):
            model_file = "{0}_{1}_{2}.txt".format(e, time_penalties[p], seeds[e][s])

            with open(log_dir + model_file) as f:
                lines = [l[:-1] for l in f.readlines()[2:]]
                tokens = [str.split(l) for l in lines]
                errors[e][p, s, :] = [float(t[1])/100.0 for t in tokens]

            with open(eval_dir + model_file) as f:
                lines = [l[:-1] for l in f.readlines()]
                tokens = [str.split(l) for l in lines]
                error_vs_diff[e][p, s, :] = [float(t[1])/100.0 for t in tokens]
                if time_penalties[p] != "x":
                    ponder_vs_diff[e][p, s, :] = [float(t[2]) for t in tokens]

mean_errors = {e: np.mean(errors[e], axis=1) for e in experiments}
mean_error_vs_diff = {e: np.mean(error_vs_diff[e], axis=1) for e in experiments}
mean_ponder_vs_diff = {e: np.mean(ponder_vs_diff[e], axis=1) for e in experiments}

last_errors = {e: errors[e][:, :, -1] for e in experiments}
last_error_means = {e: np.mean(last_errors[e], axis=1) for e in experiments}
last_error_std = {e: np.std(last_errors[e], axis=1) for e in experiments}


# error images
for i, e in enumerate(experiments):
    plt.axes().grid(True, linewidth=0.75, linestyle=":")
    x_axis = np.linspace(1, epochs * steps_per_epoch, epochs)

    for p in range(len(time_penalties)):
        for s in range(len(seeds[e])):
            plt.plot(x_axis, errors[e][p, s], color=penalty_colors[p], linewidth=1.0, alpha=0.1)
        plt.plot(x_axis, mean_errors[e][p], color=penalty_colors[p], linewidth=1.5, alpha=1.0)

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel("Iterations", fontsize=15)
    plt.ylabel("Sequence Error Rate", fontsize=15)
    plt.tick_params(left="off", bottom="off")

    plt.xlim(0, epochs * steps_per_epoch)
    plt.ylim(0, error_image_y_limits[i])

    plt.axes().set_aspect(steps_per_epoch * epochs / error_image_y_limits[i] / 1.1)
    plt.savefig(image_dir + e + "_error.png", dpi=dpi, bbox_inches='tight')

    # plt.show()
    plt.close()


# error_vs_diff images
for i, e in enumerate(experiments):
    plt.axes().grid(True, linewidth=0.75, linestyle=":")
    x_axis = np.arange(difficulties[i][0], difficulties[i][1]+1, 1)
    plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))

    for p in range(len(time_penalties)):
        for s in range(len(seeds[e])):
            plt.plot(x_axis, error_vs_diff[e][p, s], color=penalty_colors[p], linewidth=1.0, alpha=0.1)
        plt.plot(x_axis, mean_error_vs_diff[e][p], color=penalty_colors[p], linewidth=1.5, alpha=1.0)

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel("Difficulty", fontsize=15)
    plt.ylabel("Sequence Error Rate", fontsize=15)
    plt.tick_params(left="off", bottom="off")

    plt.xlim(difficulties[i][0], difficulties[i][1])
    plt.ylim(0, error_vs_diff_image_y_limits[i])

    plt.axes().set_aspect((difficulties[i][1] - difficulties[i][0]) / error_vs_diff_image_y_limits[i] / 1.25)
    plt.savefig(image_dir + e + "_difficulty_error.png", dpi=dpi, bbox_inches='tight')

    # plt.show()
    plt.close()


# ponder_vs_diff images
for i, e in enumerate(experiments):
    plt.axes().grid(True, linewidth=0.75, linestyle=":")
    x_axis = np.arange(difficulties[i][0], difficulties[i][1]+1, 1)
    plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))

    for p in range(len(time_penalties)-1):
        for s in range(len(seeds[e])):
            plt.plot(x_axis, ponder_vs_diff[e][p, s], color=penalty_colors[p], linewidth=1.0, alpha=0.1)
        plt.plot(x_axis, mean_ponder_vs_diff[e][p], color=penalty_colors[p], linewidth=1.5, alpha=1.0)

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel("Difficulty", fontsize=15)
    plt.ylabel("Ponder", fontsize=15)
    plt.tick_params(left="off", bottom="off")

    plt.xlim(difficulties[i][0], difficulties[i][1])
    plt.ylim(0, ponder_vs_diff_image_y_limits[i])

    plt.axes().set_aspect((difficulties[i][1] - difficulties[i][0]) / ponder_vs_diff_image_y_limits[i] / 1.25)
    plt.savefig(image_dir + e + "_difficulty_ponder.png", dpi=dpi, bbox_inches='tight')

    # plt.show()
    plt.close()


old_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (4, 3)

# last error images
for i, e in enumerate(experiments):
    plt.axes().grid(True, linewidth=1.0, linestyle=":")
    plt.axes().set_axisbelow(True)

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9, rotation="vertical")
    plt.xlabel("Time Penalty", fontsize=14)
    plt.ylabel("Seq. Error Rate", fontsize=14)
    plt.tick_params(left="off", bottom="off")
    plt.axes().yaxis.set_ticks(np.arange(0.0, last_error_image_y_limits[i], 0.05))

    x_labels = [str.replace(t, "x", "No ACT") for t in time_penalties]
    x_positions = np.arange(len(last_error_means[e]))
    y_values = last_error_means[e]
    y_std = last_error_std[e]
    bar_width = 0.8

    plt.axes().bar(x_positions, y_values, bar_width, color=penalty_colors, yerr=y_std)
    plt.axes().set_xticklabels([""] + x_labels)

    plt.axes().set_aspect(len(x_positions) / y_values.max() / 1.5)
    plt.savefig(image_dir + e + "_last_error.png", dpi=dpi, bbox_inches='tight')

    # plt.show()
    plt.close()

plt.rcParams["figure.figsize"] = old_figsize

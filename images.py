import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['axes.linewidth'] = 0.0
mpl.rcParams['lines.dotted_pattern'] = [1, 1]
mpl.rcParams['lines.scale_dashes'] = False


epochs = 250
steps_per_epoch = 1000

log_dir = "results/logs/"
image_dir = "results/images/"

experiments = ["parity", "logic"]
time_penalties = ["0.0001", "0.001", "0.01", "0.1", "x"]
penalty_colors = ["#7200FF", "#19E0DC", "#CDE480", "#FF1600", "#000000"]
seeds = {
    "parity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "logic": [3, 6, 8]
}

error_image_y_limits = [0.5, 0.7]
last_error_image_y_limits = [0.50, 0.30]

errors = {e: np.zeros([len(time_penalties), len(seeds[e]), epochs]) for e in experiments}


for e in experiments:
    for p in range(len(time_penalties)):
        for s in range(len(seeds[e])):
            with open(log_dir + "{0}_{1}_{2}.txt".format(e, time_penalties[p], seeds[e][s])) as f:
                lines = [l[:-1] for l in f.readlines()[2:]]
                tokens = [str.split(l) for l in lines]
                errors[e][p, s, :] = [float(t[1])/100.0 for t in tokens]

last_errors = {e: errors[e][:, :, -1] for e in experiments}
mean_errors = {e: np.mean(errors[e], axis=1) for e in experiments}
last_error_means = {e: np.mean(last_errors[e], axis=1) for e in experiments}
last_error_std = {e: np.std(last_errors[e], axis=1) for e in experiments}

x_axis = np.linspace(1, epochs * steps_per_epoch, epochs)


# error images
for i, e in enumerate(experiments):
    plt.axes().grid(True, linewidth=1.0, linestyle=":")

    for p in range(len(time_penalties)):
        for s in range(len(seeds[e])):
            plt.plot(x_axis, errors[e][p, s], color=penalty_colors[p], linewidth=1.0, alpha=0.1)
        plt.plot(x_axis, mean_errors[e][p], color=penalty_colors[p], linewidth=1.5, alpha=1.0)

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Sequence Error Rate", fontsize=14)
    plt.tick_params(left="off", bottom="off")

    plt.xlim(0, epochs * steps_per_epoch)
    plt.ylim(0, error_image_y_limits[i])

    plt.axes().set_aspect(steps_per_epoch * epochs / error_image_y_limits[i] / 1.1)
    plt.savefig(image_dir + e + "_error.png", dpi=300, bbox_inches='tight')

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
    plt.savefig(image_dir + e + "_last_error.png", dpi=300, bbox_inches='tight')

    # plt.show()
    plt.close()

plt.rcParams["figure.figsize"] = old_figsize

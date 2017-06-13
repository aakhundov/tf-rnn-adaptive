import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from act_wrapper import ACTWrapper
from act_model import ACTModel

from data import generate_parity_data


def echo(message, file):
    print(message)
    file.write(message + "\n")


SEED = 0
EVAL_SIZE = 1000

DIMENSIONS = 16
TIME_STEPS = 1
INPUT_SIZE = DIMENSIONS
NUM_CLASSES = 2
NUM_OUTPUTS = 1

NUM_HIDDEN = 128
TIME_PENALTY = 0.0001
PONDER_LIMIT = 10
LEARNING_RATE = 0.001
WITH_ACT = True


if __name__ == "__main__":

    while len(sys.argv) > 1:
        option = sys.argv[1]; del sys.argv[1]

        if option == "-seed":
            SEED = int(sys.argv[1]); del sys.argv[1]
        elif option == "-penalty":
            TIME_PENALTY = float(sys.argv[1]); del sys.argv[1]
        elif option == "-act":
            WITH_ACT = bool(int(sys.argv[1])); del sys.argv[1]
        else:
            print(sys.argv[0], ": invalid option", option)
            sys.exit(1)

    model_name = "{0}_{1}_{2}".format(
        "parity",
        TIME_PENALTY if WITH_ACT else "x",
        SEED
    )

    cell = rnn.BasicRNNCell(NUM_HIDDEN)
    if WITH_ACT:
        cell = ACTWrapper(cell, ponder_limit=PONDER_LIMIT)

    inputs = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE])
    targets = tf.placeholder(tf.int64, [None, TIME_STEPS, NUM_OUTPUTS])

    print("Creating model...")
    model = ACTModel(
        inputs, targets, TIME_STEPS, NUM_CLASSES, cell, NUM_OUTPUTS, TIME_PENALTY,
        optimizer=tf.train.AdamOptimizer(LEARNING_RATE)
    )

    log_path = "./results/evaluation/" + model_name + ".txt"
    model_path = "./results/models/" + model_name + ".ckpt"

    saver = tf.train.Saver()
    log = open(log_path, "w")

    with tf.Session() as sess:
        print("Restoring model...")
        saver.restore(sess, model_path)
        print()

        for dim in range(1, DIMENSIONS+1):
            eval_xs, eval_ys = generate_parity_data(
                EVAL_SIZE, dimensions=DIMENSIONS, fixed_parity_bits=dim, seed=54321 * dim
            )

            eval_error, eval_ponder = sess.run(
                [model.evaluation, model.ponder_steps],
                feed_dict={
                    inputs: eval_xs,
                    targets: eval_ys
                }
            )

            echo("{:d}\t{:.2f}\t{:.2f}".format(
                dim, 100 * eval_error,
                np.mean(eval_ponder)
            ), log)

        log.close()

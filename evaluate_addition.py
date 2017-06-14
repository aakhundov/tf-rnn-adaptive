import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from act_wrapper import ACTWrapper
from act_model import ACTModel

from data import generate_addition_data


def echo(message, file):
    print(message)
    file.write(message + "\n")


SEED = 0
EVAL_SIZE = 1000

MAX_DIGITS = 5
EVAL_TIME_STEPS = 3
INPUT_SIZE = MAX_DIGITS * 10
NUM_CLASSES = 11
NUM_OUTPUTS = MAX_DIGITS + 1

NUM_HIDDEN = 512
PONDER_LIMIT = 10
TIME_PENALTY = 0.0001
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
        "addition",
        TIME_PENALTY if WITH_ACT else "x",
        SEED
    )

    cell = rnn.BasicLSTMCell(NUM_HIDDEN)
    if WITH_ACT:
        cell = ACTWrapper(cell, ponder_limit=PONDER_LIMIT)

    inputs = tf.placeholder(tf.float32, [None, EVAL_TIME_STEPS, INPUT_SIZE])
    targets = tf.placeholder(tf.int64, [None, EVAL_TIME_STEPS, NUM_OUTPUTS])
    seq_length = tf.placeholder(tf.int64, [None])

    print("Creating model...")
    model = ACTModel(
        inputs, targets, EVAL_TIME_STEPS, NUM_CLASSES, cell, NUM_OUTPUTS, TIME_PENALTY,
        seq_length=seq_length, target_offset=tf.ones([tf.shape(inputs)[0]], dtype=tf.int32),
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

        for digits in range(1, MAX_DIGITS+1):
            eval_xs, eval_ys, eval_seq = generate_addition_data(
                EVAL_SIZE, min_time_steps=EVAL_TIME_STEPS, max_time_steps=EVAL_TIME_STEPS,
                max_digits=MAX_DIGITS, fixed_digits=digits, seed=54321 * digits
            )

            if WITH_ACT:
                eval_error, eval_ponder = sess.run(
                    [model.evaluation, model.ponder_steps],
                    feed_dict={
                        inputs: eval_xs,
                        targets: eval_ys,
                        seq_length: eval_seq
                    }
                )

                eval_ponder = np.ravel(eval_ponder)
                eval_ponder = eval_ponder[np.nonzero(eval_ponder)]

                echo("{:d}\t{:.2f}\t{:.2f}".format(
                    digits, 100 * eval_error,
                    np.mean(eval_ponder)
                ), log)
            else:
                eval_error = sess.run(
                    model.evaluation,
                    feed_dict={
                        inputs: eval_xs,
                        targets: eval_ys,
                        seq_length: eval_seq
                    }
                )

                echo("{:d}\t{:.2f}".format(
                    digits, 100 * eval_error
                ), log)

        log.close()

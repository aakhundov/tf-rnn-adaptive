import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from act_wrapper import ACTWrapper
from act_model import ACTModel

from data import generate_logic_data


def echo(message, file):
    print(message)
    file.write(message + "\n")


SEED = 0
TRAIN_STEPS = 250000
BATCH_SIZE = 16
VAL_SIZE = 1000

MIN_GATES = 1
MAX_GATES = 10
USED_GATES = 10
MIN_TIME_STEPS = 1
MAX_TIME_STEPS = 5
INPUT_SIZE = MAX_GATES * 10 + 2
NUM_CLASSES = 2
NUM_OUTPUTS = 1

NUM_HIDDEN = 128
PONDER_LIMIT = 10
TIME_PENALTY = 0.001
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
        "logic",
        TIME_PENALTY if WITH_ACT else "x",
        SEED
    )

    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    print(model_name)
    print()
    print("min gates", MIN_GATES)
    print("max gates", MAX_GATES)
    print("used gates", USED_GATES)
    print("min time steps", MIN_TIME_STEPS)
    print("max time steps", MAX_TIME_STEPS)
    print("time penalty", TIME_PENALTY)
    print("ponder limit", PONDER_LIMIT)
    print("learning rate", LEARNING_RATE)
    print("with ACT" if WITH_ACT else "without ACT")
    print()

    cell = rnn.BasicLSTMCell(NUM_HIDDEN)
    if WITH_ACT:
        cell = ACTWrapper(cell, ponder_limit=PONDER_LIMIT)

    inputs = tf.placeholder(tf.float32, [None, MAX_TIME_STEPS, INPUT_SIZE])
    targets = tf.placeholder(tf.int64, [None, MAX_TIME_STEPS, NUM_OUTPUTS])
    seq_length = tf.placeholder(tf.int64, [None])

    print("Creating model...")
    model = ACTModel(
        inputs, targets, MAX_TIME_STEPS, NUM_CLASSES, cell, NUM_OUTPUTS, TIME_PENALTY,
        seq_length=seq_length, optimizer=tf.train.AdamOptimizer(LEARNING_RATE)
    )

    log_path = "./results/logs/" + model_name + ".txt"
    model_path = "./results/models/" + model_name + ".ckpt"

    saver = tf.train.Saver()
    log = open(log_path, "w")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    with tf.Session(config=config) as sess:
        print("Initializing variables...")
        sess.run(tf.global_variables_initializer())

        print("Training...")
        print()

        if WITH_ACT:
            echo("{:10}{:<10}{:<15}{:<17}{:<30}".format(
                "steps", "error",
                "softmax loss", "ponder loss",
                "min / avg / std / max ponder"
            ), log)
            echo("-" * 83, log)
        else:
            echo("{:10}{:<10}{:<15}".format(
                "steps", "error", "softmax loss"
            ), log)
            echo("-" * 32, log)

        val_xs, val_ys, val_seq = generate_logic_data(
            VAL_SIZE, min_time_steps=MIN_TIME_STEPS, max_time_steps=MAX_TIME_STEPS,
            min_gates=MIN_GATES, max_gates=MAX_GATES, used_gates=USED_GATES, seed=12345)

        for step in range(TRAIN_STEPS):
            batch_xs, batch_ys, batch_seq = generate_logic_data(
                BATCH_SIZE, min_time_steps=MIN_TIME_STEPS, max_time_steps=MAX_TIME_STEPS,
                min_gates=MIN_GATES, max_gates=MAX_GATES, used_gates=USED_GATES)

            sess.run(model.training, feed_dict={
                inputs: batch_xs,
                targets: batch_ys,
                seq_length: batch_seq
            })

            if (step + 1) % 1000 == 0:
                if WITH_ACT:
                    val_error, val_soft_loss, val_pond_loss, val_ponder = sess.run(
                        [model.evaluation, model.softmax_loss, model.ponder_loss, model.ponder_steps],
                        feed_dict={
                            inputs: val_xs,
                            targets: val_ys,
                            seq_length: val_seq
                        }
                    )

                    val_ponder = np.ravel(val_ponder)
                    val_ponder = val_ponder[np.nonzero(val_ponder)]

                    echo("{:<10d}{:<10.2f}{:<15.6}{:<17.6}{:<30}".format(
                        step + 1, 100 * val_error,
                        val_soft_loss, val_pond_loss,
                        "{:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
                            np.min(val_ponder), np.mean(val_ponder),
                            np.std(val_ponder), np.max(val_ponder)
                        )
                    ), log)
                else:
                    val_error, val_loss = sess.run(
                        [model.evaluation, model.softmax_loss],
                        feed_dict={
                            inputs: val_xs,
                            targets: val_ys,
                            seq_length: val_seq
                        }
                    )

                    echo("{:<10d}{:<10.2f}{:<15.6}".format(
                        step + 1, 100 * val_error, val_loss
                    ), log)

        print()
        print("Saving model...")
        saver.save(sess, model_path)
        log.close()

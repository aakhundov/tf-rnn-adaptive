import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from act_wrapper import ACTWrapper
from data import generate_parity_data


TRAIN_STEPS = 1000000
BATCH_SIZE = 128
VAL_SIZE = 1000

TIME_STEPS = 1
INPUT_SIZE = 64
TARGET_SIZE = 2

NUM_HIDDEN = 128
TIME_PENALTY = 0.001
LEARNING_RATE = 0.0001


# setting up RNN cells
cell = rnn.BasicRNNCell(NUM_HIDDEN)
cell = ACTWrapper(cell, ponder_limit=100)

# setting up placeholders
xs = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE])
ys = tf.placeholder(tf.float32, [None, TARGET_SIZE])

# creating RNN with static_rnn()
print("Creating network...")
rnn_outputs, rnn_state = rnn.static_rnn(
    cell, tf.unstack(xs, axis=1), dtype=tf.float32
)

# inference artifacts
out_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, TARGET_SIZE], stddev=0.01), name="out_weights")
out_bias = tf.Variable(tf.constant(0.1, shape=[TARGET_SIZE]), name="out_biases")
logits = tf.matmul(rnn_outputs[-1], out_weights) + out_bias

# evaluation artifacts
mistakes = tf.not_equal(tf.argmax(ys, 1), tf.argmax(logits, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# training artifacts
softmax_loss = tf.losses.softmax_cross_entropy(ys, logits)
if isinstance(cell, ACTWrapper):
    ponder_loss = TIME_PENALTY * cell.ponder_cost
    total_loss = softmax_loss + ponder_loss
else:
    total_loss = softmax_loss
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
print("Computing gradients...")
train = optimizer.minimize(total_loss)


with tf.Session() as sess:
    print("Initializing variables...")
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    val_xs, val_ys = generate_parity_data(VAL_SIZE, dimensions=INPUT_SIZE, seed=1)

    for step in range(TRAIN_STEPS):
        batch_xs, batch_ys = generate_parity_data(BATCH_SIZE, dimensions=INPUT_SIZE)

        sess.run(train, feed_dict={
            xs: batch_xs,
            ys: batch_ys[:, 0, :]
        })

        if (step + 1) % 1000 == 0:
            if isinstance(cell, ACTWrapper):
                val_error, val_soft_loss, val_pond_loss, val_ponder = sess.run(
                    [error, softmax_loss, ponder_loss, cell.ponder_steps],
                    feed_dict={
                        xs: val_xs,
                        ys: val_ys[:, 0, :]
                    }
                )

                print(("steps {:2d}  val error {:3.2f}%  soft loss {:.6}  pond loss {:.6}  " +
                       "min {:.2f}  avg {:.2f}  std {:.2f}  max {:.2f}").format(
                        step + 1,
                        100 * val_error,
                        val_soft_loss,
                        val_pond_loss,
                        np.min(val_ponder),
                        np.average(val_ponder),
                        np.std(val_ponder),
                        np.max(val_ponder)
                    )
                )
            else:
                val_error, val_loss = sess.run(
                    [error, total_loss],
                    feed_dict={
                        xs: val_xs,
                        ys: val_ys[:, 0, :]
                    }
                )

                print("steps {:2d}  val error {:3.2f}%  val loss {:.6f}".format(
                    step + 1,
                    100 * val_error,
                    val_loss
                ))

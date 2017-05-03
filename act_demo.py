import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.examples.tutorials.mnist import input_data

from act_wrapper import ACTWrapper


DATA_FOLDER = "MNIST_data/"

EPOCHS = 20
TRAIN_STEPS = 600
BATCH_SIZE = 100

IMG_ROWS = 28
IMG_COLS = 28
IMG_LABELS = 10

TIME_STEPS = 28
TIME_OFFSET = 0
NUM_HIDDEN = 128
TIME_PENALTY = 0.01
LEARNING_RATE = 0.001


# setting up RNN cells
cell = rnn.BasicLSTMCell(NUM_HIDDEN)
cell = ACTWrapper(cell, ponder_limit=10)

# setting up placeholders
xs = tf.placeholder(tf.float32, [None, IMG_ROWS * IMG_COLS])
ys = tf.placeholder(tf.float32, [None, IMG_LABELS])

# reshaping MNIST data in a square form
images = tf.reshape(xs, [-1, IMG_ROWS, IMG_COLS])


def create_rnn_manually():
    """Create an RNN by manually chaining its unrolled layers"""
    state = cell.zero_state(tf.shape(images)[0], dtype=tf.float32)

    with tf.variable_scope("RNN"):
        for time_step in range(TIME_STEPS):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            time_step_input = images[:, time_step + TIME_OFFSET, :]
            output, state = cell(time_step_input, state)

    return output


def create_rnn_with_while_loop():
    """Create an RNN by means of tf.while_loop() function"""
    batch_size = tf.shape(images)[0]
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    with tf.variable_scope("RNN"):
        first_input = images[:, 0, :]
        first_output, first_state = cell(first_input, init_state)

        tf.get_variable_scope().reuse_variables()

        def cond(time_step, *_):
            return time_step < TIME_STEPS

        def body(time_step, output, state):
            time_step_input = images[:, time_step + TIME_OFFSET, :]
            output, state = cell(time_step_input, state)

            return time_step + 1, output, state

        last_time_step, final_output, final_state = tf.while_loop(
            cond, body, [tf.constant(1), first_output, first_state]
        )

    return final_output


def create_rnn_with_raw_rnn():
    """Create an RNN by means of tf.raw_rnn() function"""
    def loop_fn(time, cell_output, cell_state, *_):
        batch_size = tf.shape(xs)[0]

        elements_finished = tf.fill([batch_size], time >= TIME_STEPS)

        if cell_output is None:  # time == 0
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state

        next_input = images[:, tf.minimum(time, TIME_STEPS) + TIME_OFFSET, :]

        emit_output = cell_output
        next_loop_state = None

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)

    return outputs_ta.read(TIME_STEPS - 1)


def create_rnn_with_dynamic_rnn():
    """Create an RNN by means of tf.dynamic_rnn() function"""
    rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
        cell, images[:, TIME_OFFSET: TIME_OFFSET + TIME_STEPS, :], dtype=tf.float32
    )

    trans_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
    last_partition = tf.dynamic_partition(trans_outputs, [0] * (TIME_STEPS - 1) + [1], 2)[1]

    return tf.reshape(last_partition, [-1, NUM_HIDDEN])


# creating RNN and fetching its last output
# TODO: currently only create_rnn_manually() works with ACTWrapper,
# TODO: other methods cause strange error during backpropagation
print("Creating network...")
last_output = create_rnn_manually()

# inference artifacts
out_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, IMG_LABELS], stddev=0.01), name="out_weights")
out_bias = tf.Variable(tf.constant(0.1, shape=[IMG_LABELS]), name="out_biases")
prediction = tf.nn.softmax(tf.matmul(last_output, out_weights) + out_bias)

# evaluation artifacts
mistakes = tf.not_equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# training artifacts
loss = -tf.reduce_sum(ys * tf.log(prediction))
if isinstance(cell, ACTWrapper):
    loss = loss + TIME_PENALTY * cell.ponder_cost
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
print("Computing gradients...")
train = optimizer.minimize(loss)


# (downloading and) extracting MNIST dataset
print("Preparing data...")
dataset = input_data.read_data_sets(DATA_FOLDER, one_hot=True)


with tf.Session() as sess:
    print("Initializing variables...")
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    for epoch in range(EPOCHS):
        max_ponder, avg_ponder, min_ponder = [], [], []

        for step in range(TRAIN_STEPS):
            batch_xs, batch_ys = dataset.train.next_batch(BATCH_SIZE)

            sess.run(train, feed_dict={
                xs: batch_xs,
                ys: batch_ys
            })

        if isinstance(cell, ACTWrapper):
            val_error, val_ponder = sess.run(
                [error, cell.ponder_steps],
                feed_dict={
                    xs: dataset.validation.images,
                    ys: dataset.validation.labels
                }
            )

            print('Epoch {:2d}  error {:3.2f}%  min {:.2f}  avg {:.2f}  std {:.2f}  max {:.2f}'.format(
                epoch + 1,
                100 * val_error,
                np.min(val_ponder),
                np.average(val_ponder),
                np.std(val_ponder),
                np.max(val_ponder)
            ))
        else:
            val_error = sess.run(
                error,
                feed_dict={
                    xs: dataset.validation.images,
                    ys: dataset.validation.labels
                }
            )

            print('Epoch {:2d}  error {:3.2f}%'.format(
                epoch + 1,
                100 * val_error
            ))

    if isinstance(cell, ACTWrapper):
        test_error, test_ponder = sess.run(
            [error, cell.ponder_steps],
            feed_dict={
                xs: dataset.test.images,
                ys: dataset.test.labels
            }
        )

        print()
        print('Test error {:3.2f}%  min {:.2f}  avg {:.2f}  std {:.2f}  max {:.2f}'.format(
            100 * test_error,
            np.min(test_ponder),
            np.average(test_ponder),
            np.std(test_ponder),
            np.max(test_ponder)
        ))
    else:
        test_error = sess.run(
            error,
            feed_dict={
                xs: dataset.test.images,
                ys: dataset.test.labels
            }
        )

        print()
        print("Test error {:3.2f}% ".format(
            100 * test_error
        ))

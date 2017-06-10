import functools

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as s2s

from act_wrapper import ACTWrapper


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


class ACTModel:
    def __init__(self, data, target, time_steps, num_classes, rnn_cell,
                 num_outputs=1, time_penalty=0.001, seq_length=None,
                 target_offset=None, optimizer=None):

        self.data = data
        self.target = target
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.cell = rnn_cell
        self.num_outputs = num_outputs
        self.time_penalty = time_penalty
        self.seq_length = seq_length
        self.target_offset = target_offset
        self.optimizer = optimizer if optimizer \
            else tf.train.AdamOptimizer(0.001)

        self.num_hidden = rnn_cell.output_size

        self._softmax_loss = None
        self._ponder_loss = None
        self._ponder_steps = None
        self._boolean_mask = None
        self._numerical_mask = None

        if self.seq_length is not None:
            self._boolean_mask = tf.sequence_mask(self.seq_length, self.time_steps)
            if self.target_offset is not None:
                offset_mask = tf.logical_not(tf.sequence_mask(self.target_offset, self.time_steps))
                self._boolean_mask = tf.logical_and(self._boolean_mask, offset_mask)
            self._numerical_mask = tf.cast(self._boolean_mask, data.dtype)

        self.logits
        self.training
        self.evaluation

    @lazy_property
    def logits(self):
        rnn_outputs, rnn_state = rnn.static_rnn(
            self.cell, tf.unstack(self.data, axis=1),
            dtype=tf.float32
        )

        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.num_hidden])

        logits_per_output = []
        for i in range(self.num_outputs):
            initial_weights = tf.truncated_normal([self.num_hidden, self.num_classes], stddev=0.01)
            initial_biases = tf.constant(0.1, shape=[self.num_classes])
            output_weights = tf.Variable(initial_weights, name="output_weights_" + str(i))
            output_biases = tf.Variable(initial_biases, name="output_biases_" + str(i))
            logits = tf.matmul(rnn_outputs, output_weights) + output_biases
            reshaped = tf.reshape(logits, [self.time_steps, -1, self.num_classes])
            logits_per_output.append(tf.transpose(reshaped, perm=(1, 0, 2)))

        return logits_per_output

    @lazy_property
    def evaluation(self):
        mistakes_per_output = []
        for i in range(len(self.logits)):
            if self.seq_length is not None:
                mistakes = tf.reduce_any(
                    tf.logical_and(
                        tf.not_equal(self.target[:, :, i], tf.argmax(self.logits[i], 2)),
                        self._boolean_mask
                    ), axis=1
                )
            else:
                mistakes = tf.reduce_any(
                    tf.not_equal(self.target[:, :, i], tf.argmax(self.logits[i], 2)), axis=1
                )
            mistakes_per_output.append(mistakes)

        if len(mistakes_per_output) == 1:
            all_mistakes = mistakes_per_output[0]
        else:
            stacked_mistakes = tf.stack(mistakes_per_output)
            all_mistakes = tf.reduce_any(stacked_mistakes, axis=0)

        return tf.reduce_mean(tf.cast(all_mistakes, tf.float32))

    @lazy_property
    def training(self):
        softmax_loss_per_output = []
        for i in range(len(self.logits)):
            if self.seq_length is not None:
                softmax_loss = s2s.sequence_loss(
                    self.logits[i], self.target[:, :, i], self._numerical_mask
                )
            else:
                softmax_loss = s2s.sequence_loss(
                    self.logits[i], self.target[:, :, i],
                    tf.ones_like(self.target[:, :, i], self.logits[i].dtype)
                )
            softmax_loss_per_output.append(softmax_loss)

        if len(softmax_loss_per_output) == 1:
            self._softmax_loss = softmax_loss_per_output[0]
        else:
            self._softmax_loss = tf.add_n(softmax_loss_per_output)

        if isinstance(self.cell, ACTWrapper):
            self._ponder_loss = self.time_penalty * self.cell.get_ponder_cost(self.seq_length)
            self._ponder_steps = self.cell.get_ponder_steps(self.seq_length)
            total_loss = self._softmax_loss + self._ponder_loss
        else:
            total_loss = self._softmax_loss

        return self.optimizer.minimize(total_loss)

    @lazy_property
    def softmax_loss(self):
        return self._softmax_loss

    @lazy_property
    def ponder_loss(self):
        if isinstance(self.cell, ACTWrapper):
            return self._ponder_loss
        else:
            raise TypeError("ACT wrapper is not used")

    @lazy_property
    def ponder_steps(self):
        if isinstance(self.cell, ACTWrapper):
            return self._ponder_steps
        else:
            raise TypeError("ACT wrapper is not used")

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope


class ACTWrapper(rnn.RNNCell):
    """Adaptive Computation Time wrapper (based on https://arxiv.org/abs/1603.08983)"""

    def __init__(self, cell, ponder_limit=100, epsilon=0.01, init_halting_bias=1.0, reuse=None):
        self._cell = cell
        self._ponder_limit = ponder_limit
        self._epsilon = epsilon
        self._init_halting_bias = init_halting_bias
        self._reuse = reuse

        self._ponder_steps_op = None
        self._ponder_cost_op = None

        self._ponder_steps = []
        self._remainders = []

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def ponder_steps(self):
        if len(self._ponder_steps) == 0:
            raise RuntimeError("ponder_steps should be invoked after all call()'s")
        if self._ponder_steps_op is None:
            self._ponder_steps_op = tf.stack(self._ponder_steps)
        return self._ponder_steps_op

    @property
    def ponder_cost(self):
        if len(self._remainders) == 0:
            raise RuntimeError("ponder_cost should be invoked after all call()'s")
        if self._ponder_cost_op is None:
            batch_size = tf.cast(tf.shape(self._remainders[0])[0], self._remainders[0].dtype)
            self._ponder_cost_op = tf.reduce_sum(tf.stack(self._remainders)) / batch_size
        return self._ponder_cost_op

    def __call__(self, inputs, state, scope=None):
        with _checked_scope(self, scope or "act_wrapper", reuse=self._reuse):
            batch_size = tf.shape(inputs)[0]
            if isinstance(state, tuple):
                state_is_tuple = True
                state_tuple_type = type(state)
            else:
                state_is_tuple = False
                state_tuple_type = None

            inputs_and_zero = tf.concat([inputs, tf.fill([batch_size, 1], 0.0)], 1)
            inputs_and_one = tf.concat([inputs, tf.fill([batch_size, 1], 1.0)], 1)
            zero_state = tf.convert_to_tensor(self._cell.zero_state(batch_size, state.dtype))
            zero_output = tf.fill([batch_size, self._cell.output_size], tf.constant(0.0, state.dtype))

            def cond(finished, *_):
                return tf.reduce_any(tf.logical_not(finished))

            def body(previous_finished, time_step,
                     previous_state, running_output, running_state,
                     ponder_steps, remainders, running_p_sum):

                current_inputs = tf.where(tf.equal(time_step, 1), inputs_and_one, inputs_and_zero)
                current_output, current_state = self._cell(current_inputs, previous_state)

                if state_is_tuple:
                    joint_current_state = tf.concat(current_state, 1)
                else:
                    joint_current_state = current_state

                current_h = tf.nn.sigmoid(tf.squeeze(
                    _linear([joint_current_state], 1, True, self._init_halting_bias), 1
                ))

                current_h_sum = running_p_sum + current_h

                limit_condition = time_step >= self._ponder_limit
                halting_condition = current_h_sum >= 1.0 - self._epsilon
                current_finished = tf.logical_or(halting_condition, limit_condition)
                just_finished = tf.logical_xor(current_finished, previous_finished)

                current_p = tf.where(current_finished, 1.0 - running_p_sum, current_h)
                expanded_current_p = tf.expand_dims(current_p, 1)

                running_output += expanded_current_p * current_output

                if state_is_tuple:
                    running_state += tf.expand_dims(expanded_current_p, 0) * current_state
                else:
                    running_state += expanded_current_p * current_state

                ponder_steps = tf.where(just_finished, tf.fill([batch_size], time_step), ponder_steps)
                remainders = tf.where(just_finished, current_p, remainders)
                running_p_sum += current_p

                return (current_finished, time_step + 1,
                        current_state, running_output, running_state,
                        ponder_steps, remainders, running_p_sum)

            _, _, _, final_output, final_state, all_ponder_steps, all_remainders, _ = \
                tf.while_loop(cond, body, [
                    tf.fill([batch_size], False), tf.constant(1), state, zero_output, zero_state,
                    tf.fill([batch_size], 0), tf.fill([batch_size], 0.0), tf.fill([batch_size], 0.0)
                ])

            if state_is_tuple:
                final_state = state_tuple_type(
                    *tf.unstack(final_state)
                )

            self._ponder_steps.append(all_ponder_steps)
            self._remainders.append(all_remainders)

            return final_output, final_state

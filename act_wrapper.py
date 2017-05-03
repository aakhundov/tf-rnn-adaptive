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
            self._ponder_cost_op = tf.reduce_sum(tf.stack(self._remainders))
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

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:
                    next_input = inputs_and_one
                    next_cell_state = state

                    elements_finished = tf.fill([batch_size], False)

                    next_loop_state = [
                        zero_state,                  # running state
                        zero_output,                 # running output
                        tf.fill([batch_size], 0),    # ponder steps
                        tf.fill([batch_size], 0.0),  # remainders
                        tf.fill([batch_size], 0.0)   # running sum of halting probs
                    ]
                else:
                    next_input = inputs_and_zero
                    next_cell_state = cell_state

                    if state_is_tuple:
                        joint_state = tf.concat(cell_state, 1)
                    else:
                        joint_state = cell_state

                    current_h = tf.nn.sigmoid(tf.squeeze(
                        _linear([joint_state], 1, True, self._init_halting_bias), 1
                    ))

                    previous_p_sum = loop_state[4]
                    current_h_sum = previous_p_sum + current_h

                    halting_condition = current_h_sum >= 1.0 - self._epsilon
                    previous_halting_condition = previous_p_sum >= 1.0 - self._epsilon
                    limit_condition = time >= self._ponder_limit

                    elements_finished = tf.logical_or(halting_condition, limit_condition)
                    just_finished = tf.logical_xor(elements_finished, previous_halting_condition)
                    current_p = tf.where(elements_finished, 1.0 - previous_p_sum, current_h)
                    expanded_current_p = tf.expand_dims(current_p, 1)

                    if state_is_tuple:
                        loop_state[0] += tf.expand_dims(expanded_current_p, 0) * cell_state
                    else:
                        loop_state[0] += expanded_current_p * cell_state

                    loop_state[1] += expanded_current_p * cell_output
                    loop_state[2] = tf.where(just_finished, tf.fill([batch_size], time), loop_state[2])
                    loop_state[3] = tf.where(just_finished, current_p, loop_state[3])
                    loop_state[4] = previous_p_sum + current_p

                    next_loop_state = loop_state

                return (elements_finished, next_input, next_cell_state,
                        emit_output, next_loop_state)

            _, _, last_loop_state = tf.nn.raw_rnn(self._cell, loop_fn, scope=scope)

            if state_is_tuple:
                final_state = state_tuple_type(*tf.unstack(last_loop_state[0]))
            else:
                final_state = last_loop_state[0]

            final_output = last_loop_state[1]
            ponder_steps = last_loop_state[2]
            remainders = last_loop_state[3]

            self._ponder_steps.append(ponder_steps)
            self._remainders.append(remainders)

            return final_output, final_state

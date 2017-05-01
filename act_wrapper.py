import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope


class ACTWrapper(rnn.RNNCell):
    """Adaptive Computation Time wrapper (cf. "Adaptive Computation Time for RNN's" by Alex Graves)"""

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

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output

                if cell_output is None:
                    next_input = inputs_and_one
                    next_cell_state = state

                    elements_finished = tf.fill([batch_size], False)

                    next_loop_state = [
                        tf.TensorArray(state.dtype, size=1, dynamic_size=True),   # halting probs from ponder steps
                        tf.TensorArray(state.dtype, size=1, dynamic_size=True),   # cell states from ponder steps
                        tf.fill([batch_size], 0.0)                                # running sum of halting probs
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

                    previous_p_sum = loop_state[2]
                    current_h_sum = previous_p_sum + current_h

                    halting_condition = current_h_sum >= 1.0 - self._epsilon
                    limit_condition = time >= self._ponder_limit
                    elements_finished = tf.logical_or(halting_condition, limit_condition)
                    current_p = tf.where(elements_finished, 1.0 - previous_p_sum, current_h)

                    loop_state[0] = loop_state[0].write(time - 1, current_p)
                    loop_state[1] = loop_state[1].write(time - 1, cell_state)
                    loop_state[2] = previous_p_sum + current_p
                    next_loop_state = loop_state

                return (elements_finished, next_input, next_cell_state,
                        emit_output, next_loop_state)

            outputs_ta, _, last_loop_state = tf.nn.raw_rnn(self._cell, loop_fn, scope=scope)

            ps = last_loop_state[0].stack()
            expanded_ps = tf.expand_dims(ps, 2)
            states = last_loop_state[1].stack()
            outputs = outputs_ta.stack()

            final_output = tf.reduce_sum(outputs * expanded_ps, 0)

            if state_is_tuple:
                twice_expanded_ps = tf.expand_dims(expanded_ps, 1)
                final_state_merged = tf.reduce_sum(states * twice_expanded_ps, 0)
                final_state_tuple = tf.unstack(final_state_merged, axis=0)
                final_state = state_tuple_type(*final_state_tuple)
            else:
                final_state = tf.reduce_sum(states * expanded_ps, 0)

            ponder_steps = tf.cast(tf.reduce_sum(tf.sign(ps), 0), tf.int32)
            remainder_indices = tf.stack([ponder_steps - 1, tf.range(0, batch_size)], 1)
            remainders = tf.gather_nd(ps, remainder_indices)

            self._ponder_steps.append(ponder_steps)
            self._remainders.append(remainders)

            return final_output, final_state

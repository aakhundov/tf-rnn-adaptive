import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope


class TimesWrapper(rnn.RNNCell):
    """Simple RNN wrapper that repeats each "cell" step "times" times"""

    def __init__(self, cell, times=1, reuse=None):
        self._cell = cell
        self._times = times
        self._reuse = reuse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with _checked_scope(self, scope or "act_wrapper", reuse=self._reuse):
            for time in range(self._times):
                if time > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = self._cell(inputs, state)
        return output, state

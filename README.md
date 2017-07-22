Adaptive Computation Time for RNNs
==================================

Implementation of **Adaptive Computation Time** (ACT) algorithm for recurrent neural networks proposed in ["Adaptive Computation Time for Recurrent Neural Networks"](https://arxiv.org/abs/1603.08983) paper by Alex Graves.

* **act_wrapper.py** file contains the ACTWrapper class, which can be wrapped around different TensorFlow RNN cell instances (such as BasicRNNCell, LSTMCell, GRUCell, etc.) to obtain ACT functionality on top of them.

* **act_model.py** file contains configurable TensorFlow model facilitating training RNN's with and without ACT for solving sequence labelling task. The model is general enough to be used in all **train_\*.py** files with different configuration for reproducing four experiments from the original paper.

* **train_\*.py** files reproduce four experiments from the above-mentioned paper: "parity", "logic", "addition", and "sort". The configuration parameters of each experiment are specified as upper-case constants in the beginning of each script. The default values of the parameters are downsized for reproducing on a moderate hardware but can be easily restored to ones specified in the paper. Some of the train models (in a form of TF checkpoints) and corresponding training logs are available in **results** folder.

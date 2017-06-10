Implementation of **Adaptive Computation Time** (ACT) algorithm for recurrent neural networks proposed in "Adaptive Computation Time for Recurrent Neural Networks" paper by Alex Graves (https://arxiv.org/abs/1603.08983).

**act_wrapper.py** file contains the ACTWrapper class, which can be wrapped around different TensorFlow RNN cell instances (such as BasicRNNCell, LSTMCell, GRUCell, etc.) to obtain ACT functionality on top of them.

**act_model.py** file contains configurable TensorFlow model facilitating training RNN's with and without ACT for solving sequence labelling task. The model is used in all **train_xxx.py** files with different configuration for reproducing four experiments from the original paper.

**mnist_demo.py** file contains demonstration of the ACTWrapper in application to MNIST digits recognition task: here consecutive rows of each image are fed to the RNN (ACTWrapper wrapped around TensorFlow RNN cell) at subsequent time steps.

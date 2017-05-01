Implementation of *Adaptive Computation Time* algorithm for recurrent neural networks proposed in "Adaptive Computation Time for Recurrent Neural Networks" paper by Alex Graves (https://arxiv.org/abs/1603.08983).

**act_wrapper.py** contains an ACTWrapper class that can wrapped around different TensorFlow RNN cell instances (BasicRNNCell, BasicLSTMCell, GRUCell, etc.). **act_test.py** contains demonstration of ACTWrapper in application to MNIST digits recognition task: consecutive rows of each image are fed to an RNN at subsequent time steps.

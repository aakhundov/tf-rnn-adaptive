import numpy as np


logic_gates = [
    [[1, 0], [0, 0]],
    [[0, 1], [0, 0]],
    [[0, 0], [1, 0]],
    [[0, 1], [1, 0]],
    [[1, 1], [1, 0]],
    [[0, 0], [0, 1]],
    [[1, 0], [0, 1]],
    [[1, 1], [0, 1]],
    [[1, 0], [1, 1]],
    [[0, 1], [1, 1]]
]


def generate_data(batch_size, time_steps, max_gates=10):
    inputs, targets = [], []
    gates_num = len(logic_gates)

    for b in range(batch_size):
        input_steps = np.zeros([time_steps, max_gates * gates_num + 2])
        target_steps = np.zeros([time_steps, 1])

        b0 = np.random.randint(2)
        for t in range(time_steps):
            b1 = np.random.randint(2)
            if t == 0:
                input_steps[t][0] = b0
            input_steps[t][1] = b1

            num_gates = np.random.randint(1, max_gates + 1)

            for g in range(num_gates):
                gate = np.random.randint(gates_num)
                b1, b0 = logic_gates[gate][b1][b0], b1
                input_steps[t][2 + g * gates_num + gate] = 1

            target_steps[t][0] = b1
            b0 = b1

        inputs.append(input_steps)
        targets.append(target_steps)

    return np.stack(inputs), np.stack(targets)


def test_data(inputs, targets):
    gates_num = len(logic_gates)
    for b in range(len(inputs)):
        for t in range(len(inputs[b])):
            b0 = int(inputs[b][0][0]) if t == 0 else b1
            b1 = int(inputs[b][t][1])

            if t > 0:
                assert (inputs[b][t][0] == 0.0), \
                    "First bit is not zero in batch {0} time-step {1}".format(b, t)

            for g in range((len(inputs[b][t]) - 2) // gates_num):
                one_hot = inputs[b][t][g * gates_num + 2:(g+1) * gates_num + 2]
                if np.max(one_hot) == 1.0:
                    gate = np.argmax(one_hot)
                    b1, b0 = logic_gates[gate][b1][b0], b1

            target_bit = int(targets[b][t])
            assert (target_bit == b1), \
                "Target bit does not match at batch {0} time step {1}: {2} (computed) vs. {3} (target)".format(
                    b, t, b1, target_bit
                )


for i in range(100):
    test_data(*generate_data(16, 10))

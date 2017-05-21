import numpy as np


def generate_data(batch_size, dimensions=64, seed=None):
    inputs, targets = [], []

    if seed is not None:
        np.random.seed(seed)

    for b in range(batch_size):
        vector = np.zeros([dimensions])
        idx = np.arange(0, dimensions)
        np.random.shuffle(idx)

        threshold = np.random.randint(1, dimensions + 1)
        values = np.random.randint(2, size=[threshold]) * 2 - 1
        vector[idx[:threshold]] = values

        inputs.append([vector])
        targets.append([[
            np.sum(np.where(
                values > 0, [1.0], [0.0]
            )) % 2
        ]])

    if seed is not None:
        np.random.seed()

    return np.stack(inputs), np.stack(targets)


def test_data(inputs, targets):
    for b in range(len(inputs)):
        computed_parity = 0.0
        for d in range(len(inputs[b][0])):
            if inputs[b][0][d] == 1.0:
                computed_parity = 1 - computed_parity

        target_parity = int(targets[b][0][0])
        assert (computed_parity == target_parity),\
            "Parity does not match at batch {0}: {1} (computed) vs. {2} (target)".format(
                    b, computed_parity, target_parity
                )


for i in range(100):
    test_data(*generate_data(128))

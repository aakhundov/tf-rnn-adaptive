import numpy as np


def generate_data(batch_size, min_numbers=2, max_numbers=15, seed=None):
    inputs, targets, seq_length = [], [], []

    if seed is not None:
        np.random.seed(seed)

    for b in range(batch_size):
        input_steps = np.zeros([max_numbers * 2, 2])
        target_steps = np.zeros([max_numbers * 2, max_numbers])

        numbers_count = np.random.randint(
            min_numbers,
            max_numbers + 1
        )
        seq_length.append(numbers_count * 2)
        numbers = np.random.randn(numbers_count)
        indices = np.argsort(numbers)

        for n in range(numbers_count * 2):
            if n < numbers_count:
                input_steps[n][1] = numbers[n]
                if n == numbers_count - 1:
                    input_steps[n][0] = 1
            else:
                target_steps[n][indices[n - numbers_count]] = 1

        inputs.append(input_steps)
        targets.append(target_steps)

    if seed is not None:
        np.random.seed()

    return np.stack(inputs), np.stack(targets), seq_length


def test_data(inputs, targets, seq_length):
    for b in range(len(inputs)):
        numbers, target_indices = [], []
        numbers_count = seq_length[b] // 2

        for n in range(numbers_count):
            numbers.append(inputs[b][n][1])
            target_indices.append(
                np.argmax(targets[b][numbers_count + n])
            )

            if n == numbers_count - 1:
                assert (inputs[b][n][0] == 1.0), "End bit is not one at batch {0} time step {1}".format(b, n)
            else:
                assert (inputs[b][n][0] == 0.0), "End bit is not zero at batch {0} time step {1}".format(b, n)

        computed_indices = np.argsort(numbers)
        assert np.all(computed_indices == target_indices), \
            "Sorted indices don't match at batch {0}: {1} (computed) vs. {2} (target)".format(
                b, computed_indices, target_indices
            )


for i in range(100):
    test_data(*generate_data(16))

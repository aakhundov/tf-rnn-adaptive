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


def generate_parity_data(batch_size, dimensions=64, fixed_parity_bits=0, seed=None):
    inputs, targets = [], []

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    for b in range(batch_size):
        input_vector = np.zeros([dimensions])

        idx = np.arange(0, dimensions)
        np.random.shuffle(idx)

        if fixed_parity_bits > 0 and fixed_parity_bits <= dimensions:
            threshold = fixed_parity_bits
        else:
            threshold = np.random.randint(1, dimensions + 1)

        values = np.random.randint(2, size=[threshold]) * 2 - 1
        parity = np.sum(np.where(values > 0, [1.0], [0.0]), dtype=np.int32) % 2
        input_vector[idx[:threshold]] = values

        inputs.append([input_vector])
        targets.append([[parity]])

    if seed is not None:
        np.random.set_state(state)

    return np.stack(inputs), np.stack(targets)


def generate_logic_data(batch_size, min_time_steps=1, max_time_steps=10,
                        min_gates=1, max_gates=10, fixed_gates=0, used_gates=10, seed=None):
    inputs, targets, seq_length = [], [], []

    if used_gates > len(logic_gates):
        raise AttributeError("used_gates can't be greater than {0}".format(len(logic_gates)))

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    for b in range(batch_size):
        input_steps = np.zeros([max_time_steps, max_gates * len(logic_gates) + 2])
        target_steps = np.zeros([max_time_steps, 1])

        seq_length.append(
            np.random.randint(
                min_time_steps,
                max_time_steps + 1
            )
        )

        b0 = np.random.randint(2)
        for t in range(seq_length[-1]):
            b1 = np.random.randint(2)
            if t == 0:
                input_steps[t][0] = b0
            input_steps[t][1] = b1

            if fixed_gates > 0 and fixed_gates <= max_gates:
                num_gates = fixed_gates
            else:
                num_gates = np.random.randint(min_gates, max_gates + 1)

            for g in range(num_gates):
                gate = np.random.randint(used_gates)
                b1, b0 = logic_gates[gate][b1][b0], b1
                input_steps[t][2 + g * len(logic_gates) + gate] = 1

            target_steps[t][0] = b1
            b0 = b1

        inputs.append(input_steps)
        targets.append(target_steps)

    if seed is not None:
        np.random.set_state(state)

    return np.stack(inputs), np.stack(targets), seq_length


def generate_addition_data(batch_size, min_time_steps=1, max_time_steps=5, max_digits=5,
                           fixed_digits=0, seed=None):

    inputs, targets, seq_length = [], [], []

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    for b in range(batch_size):
        input_steps = np.zeros([max_time_steps, max_digits * 10])
        target_steps = np.zeros([max_time_steps, max_digits + 1])

        seq_length.append(
            np.random.randint(
                min_time_steps,
                max_time_steps + 1
            )
        )

        running_sum = 0
        for t in range(seq_length[-1]):
            if fixed_digits > 0 and fixed_digits <= max_digits:
                digits_no = fixed_digits
            else:
                digits_no = np.random.randint(1, max_digits + 1)

            current_number = np.random.randint(10 ** (digits_no - 1), 10 ** digits_no)
            number_digits = np.array([int(c) for c in str(current_number)])

            for d in range(digits_no):
                input_steps[t][d * 10 + number_digits[d]] = 1

            running_sum += current_number
            sum_digits = np.array([int(c) for c in str(running_sum)])

            for d in range(max_digits + 1):
                if d < len(sum_digits):
                    target_steps[t][d] = sum_digits[d]
                else:
                    target_steps[t][d] = 10

        inputs.append(input_steps)
        targets.append(target_steps)

    if seed is not None:
        np.random.set_state(state)

    return np.stack(inputs), np.stack(targets), seq_length


def generate_sort_data(batch_size, min_numbers=2, max_numbers=15, fixed_numbers=0, seed=None):
    inputs, targets, seq_length = [], [], []

    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)

    for b in range(batch_size):
        input_steps = np.zeros([max_numbers * 2, 2])
        target_steps = np.zeros([max_numbers * 2, 1])

        if fixed_numbers > 0 and fixed_numbers <= max_numbers:
            numbers_count = fixed_numbers
        else:
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
                target_steps[n][0] = indices[n - numbers_count]

        inputs.append(input_steps)
        targets.append(target_steps)

    if seed is not None:
        np.random.set_state(state)

    return np.stack(inputs), np.stack(targets), seq_length


def test_parity_data(inputs, targets):
    for b in range(len(inputs)):
        computed_parity = 0
        for d in range(len(inputs[b][0])):
            if inputs[b][0][d] == 1.0:
                computed_parity = 1 - computed_parity

        target_parity = targets[b][0][0]
        assert (computed_parity == target_parity),\
            "Parity does not match at batch {0}: {1} (computed) vs. {2} (target)".format(
                    b, computed_parity, target_parity
                )


def test_logic_data(inputs, targets, seq_length):
    gates_num = len(logic_gates)
    for b in range(len(inputs)):
        for t in range(seq_length[b]):
            b0 = int(inputs[b][0][0]) if t == 0 else b1
            b1 = int(inputs[b][t][1])

            if t > 0:
                assert (inputs[b][t][0] == 0.0), \
                    "First bit is not zero in batch {0} time-step {1}".format(b, t)

            for g in range((len(inputs[b][t]) - 2) // gates_num):
                one_hot = inputs[b][t][g * gates_num + 2:(g+1) * gates_num + 2]
                if np.max(one_hot) == 1.0:
                    gate = np.asscalar(np.argmax(one_hot))
                    b1, b0 = logic_gates[gate][b1][b0], b1

            target_bit = targets[b][t][0]
            assert (target_bit == b1), \
                "Target bit does not match at batch {0} time step {1}: {2} (computed) vs. {3} (target)".format(
                    b, t, b1, target_bit
                )


def test_addition_data(inputs, targets, seq_length):
    for b in range(len(inputs)):
        running_sum = 0
        for t in range(seq_length[b]):
            current_number = 0
            for d in range(len(inputs[b][t]) // 10):
                one_hot = inputs[b][t][d * 10:d * 10 + 10]
                if np.max(one_hot) == 1.0:
                    current_number = current_number * 10 + np.argmax(one_hot)

            target_sum = 0
            for d in range(len(targets[b][t])):
                if targets[b][t][d] != 10:
                    target_sum = target_sum * 10 + targets[b][t][d]

            running_sum += current_number
            assert (running_sum == target_sum), \
                "Running sum doesn't match at batch {0} time step {1}: {2} (computed) vs. {3} (target)".format(
                    b, t, running_sum, target_sum
                )


def test_sort_data(inputs, targets, seq_length):
    for b in range(len(inputs)):
        numbers, target_indices = [], []
        numbers_count = seq_length[b] // 2

        for n in range(numbers_count):
            numbers.append(inputs[b][n][1])
            target_indices.append(targets[b][numbers_count + n][0])

            if n == numbers_count - 1:
                assert (inputs[b][n][0] == 1.0), "End bit is not one at batch {0} time step {1}".format(b, n)
            else:
                assert (inputs[b][n][0] == 0.0), "Non-end bit is not zero at batch {0} time step {1}".format(b, n)

        computed_indices = np.argsort(numbers)
        assert np.all(computed_indices == target_indices), \
            "Sorted indices don't match at batch {0}: {1} (computed) vs. {2} (target)".format(
                b, computed_indices, target_indices
            )


if __name__ == "__main__":

    print("Testing parity data... ", end="")
    for i in range(100):
        test_parity_data(*generate_parity_data(128, fixed_parity_bits=i % 10))
    print("passed")

    print("Testing logic data... ", end="")
    for i in range(100):
        test_logic_data(*generate_logic_data(16, fixed_gates=i % 10))
    print("passed")

    print("Testing addition data... ", end="")
    for i in range(100):
        test_addition_data(*generate_addition_data(32, fixed_digits=i % 10))
    print("passed")

    print("Testing sort data... ", end="")
    for i in range(100):
        test_sort_data(*generate_sort_data(16, fixed_numbers=i % 10))
    print("passed")

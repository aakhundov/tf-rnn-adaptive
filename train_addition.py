import numpy as np


def generate_data(batch_size, time_steps, max_digits=5):
    inputs, targets = [], []

    for b in range(batch_size):
        input_steps = np.zeros([time_steps, max_digits * 10])
        target_steps = np.zeros([time_steps, max_digits+1, 11])

        running_sum = 0
        for t in range(time_steps):
            digits_no = np.random.randint(1, max_digits + 1)
            current_number = np.random.randint(10 ** (digits_no - 1), 10 ** digits_no)
            number_digits = np.array([int(c) for c in str(current_number)])

            for d in range(digits_no):
                input_steps[t][d * 10 + number_digits[d]] = 1

            running_sum += current_number
            sum_digits = np.array([int(c) for c in str(running_sum)])

            for d in range(max_digits + 1):
                if d < len(sum_digits):
                    target_steps[t][d][sum_digits[d]] = 1
                else:
                    target_steps[t][d][10] = 1

        inputs.append(input_steps)
        targets.append(target_steps)

    return np.stack(inputs), np.stack(targets)


def test_data(inputs, targets):
    for b in range(len(inputs)):
        running_sum = 0
        for t in range(len(inputs[b])):
            current_number = 0
            for d in range(len(inputs[b][t]) // 10):
                one_hot = inputs[b][t][d * 10:d * 10 + 10]
                if np.max(one_hot) == 1.0:
                    current_number = current_number * 10 + np.argmax(one_hot)

            target_sum = 0
            for d in range(len(targets[b][t])):
                one_hot = targets[b][t][d]
                if one_hot[-1] == 0.0:
                    target_sum = target_sum * 10 + np.argmax(one_hot)

            running_sum += current_number
            assert (running_sum == target_sum), \
                "Running sum doesn't match at batch {0} time step {1}: {2} (computed) vs. {3} (target)".format(
                    b, t, running_sum, target_sum
                )


for i in range(100):
    test_data(*generate_data(32, 5))

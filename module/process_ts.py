import numpy as np


def process_ts(x_ts, interval_len, train_size, noise_length, horizon=1, test_size=0):
    x_ts = x_ts.T

    N, n_ts = x_ts.shape

    print(N, n_ts)

    val_start = train_size + interval_len + 1
    val_size = noise_length - val_start - test_size
    test_start = train_size + interval_len + 1 + val_size

    def rolling_windows(start, num_win):

        x = np.zeros((num_win, n_ts, interval_len))
        y = np.zeros((num_win, n_ts, horizon))

        x = np.zeros((num_win, interval_len))
        y = np.zeros((num_win, horizon))

        for j in range(start, start + num_win):
            x[j - start, ...] = x_ts[j - interval_len: j, :][::-1, :].T

            y[j - start, ...] = x_ts[j: j + horizon, :].T

        return x, y

    data_ts, y_ts = rolling_windows(interval_len, train_size)
    if val_size > 0:
        val_ts, y_val = rolling_windows(val_start, val_size)
    else:
        val_ts, y_val = None, None

    if test_size > 0:
        test_ts, y_test = rolling_windows(test_start, test_size)
    else:
        test_ts, y_test = None, None

    return data_ts, y_ts, val_ts, y_val, test_ts, y_test


def process_ts_slow(x_ts, interval_len, train_size, noise_length, **conf):

    n_ts = x_ts.shape[0]

    return_only_ts = conf.get('return_only_ts', 1)
    horizon = conf.get('horizon', 1) # prediction horizon

    test_flag = conf.get('test_flag', 0) # if validation flag = 1, noise_length is assumed to be extended by test_size

    test_size = conf.get('test_size', 0)


    data_ts = np.zeros((1, interval_len))  # first is because we got noise_length segments of 1 jumps

    val_ts = np.zeros((1, interval_len))

    y_ts = np.zeros(horizon)
    y_val = np.zeros(horizon)

    if test_flag == 1:
        test_ts = np.zeros((1, interval_len))
        y_test = np.zeros(horizon)

    # n_jumps = noise_length - 1 - interval_len
    # is because we need 1 space for y

    n_jumps = train_size

    val_start = train_size + interval_len + 1 # Align with RC, begin validate at the one AFTER washout(interval_len) + train_size
    val_size = noise_length - val_start - test_size

    if test_flag == 1:
        test_start = train_size + interval_len + 1 + val_size
        test_pieces = []
        y_test_pieces = []

    ts_pieces = []
    y_pieces = []

    val_pieces = []
    y_val_pieces = []


    # print('n_jumps = ' + str(n_jumps))

    if n_jumps > 0:
        for i in range(n_ts):
            target = x_ts[i]  # Time series
            data_ts_temp = np.zeros((n_jumps, interval_len))
            y_ts_temp = np.zeros((n_jumps, horizon))

            val_ts_temp = np.zeros((val_size, interval_len))
            y_val_temp = np.zeros((val_size, horizon))

            test_ts_temp = np.zeros((test_size, interval_len))
            y_test_temp = np.zeros((test_size, horizon))

            for j in range(n_jumps):
                # x part of time series

                temp = target[j: j + interval_len][::-1]
                # temp = temp[::-1]
                data_ts_temp[j] = temp

                # y part of time series
                # y_ts_temp[j] = target[j + interval_len]
                prediction_start = j + interval_len
                prediction_end = prediction_start + horizon
                y_ts_temp[j] = target[prediction_start: prediction_end]

            for k in range(val_start, noise_length - test_size):
                r_val = range(k - interval_len, k)

                temp_val = target[r_val][::-1]
                # temp_val = temp_val[::-1]
                #
                val_ts_temp[k - val_start] = temp_val

                # print(k)
                y_val_temp[k - val_start] = target[k: k+horizon]

            if test_flag == 1:
                for l in range(test_start, noise_length):
                    r_test = range(l - interval_len, l)

                    temp_test = target[r_test][::-1]
                    # temp_val = temp_val[::-1]
                    #
                    test_ts_temp[l - test_start] = temp_test

                    # print(k)
                    y_test_temp[l - test_start] = target[l: l + horizon]

                test_pieces.append(test_ts_temp)
                y_test_pieces.append(y_test_temp)

                test_ts = np.append(test_ts, test_ts_temp, axis=0)
                # y_val = np.append(y_val, y_val_temp, axis=0)
                y_test = np.vstack([y_test, y_test_temp])

            ts_pieces.append(data_ts_temp)
            y_pieces.append(y_ts_temp)

            val_pieces.append(val_ts_temp)
            y_val_pieces.append(y_val_temp)

            data_ts = np.append(data_ts, data_ts_temp, axis=0)
            # y_ts = np.append(y_ts, y_ts_temp, axis=0)
            y_ts = np.vstack([y_ts, y_ts_temp])

            val_ts = np.append(val_ts, val_ts_temp, axis=0)
            # y_val = np.append(y_val, y_val_temp, axis=0)
            y_val = np.vstack([y_val, y_val_temp])
    else:
        print('Interval too large for noise_length')

    y_ts = np.delete(y_ts, 0, 0)
    # y_ts = y_ts.reshape(-1)
    y_val = np.delete(y_val, 0, 0)
    # y_val = y_val.reshape(-1)

    data_ts = np.delete(data_ts, 0, 0)  # remove first place holder row.
    val_ts = np.delete(val_ts, 0, 0)  # remove first place holder row.

    if test_flag == 1:
        y_test = np.delete(y_test, 0, 0)
        test_ts = np.delete(test_ts, 0, 0)  # remove first place holder row.
    # return data_ts, y_ts, y_val, val_ts, y_test, test_ts,
    if return_only_ts == 1:
        return data_ts, y_ts
    else:
        if test_flag == 1:
            return ts_pieces, y_pieces, val_pieces, y_val_pieces, test_pieces, y_test_pieces
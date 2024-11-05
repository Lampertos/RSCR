import numpy as np
import pandas as pd


def give_ts(name, ts_index=1, noise_length=300, n_ts=1, sigma=1):
    start = 1

    # if name == 'mg':
    #     x_mg = np.loadtxt('data_/MackeyGlass_t17.txt')
    #     for i in range(5):
    #         x_mg = np.append(x_mg, x_mg)
    #     x_ts = np.zeros((n_ts, noise_length))

    #     for i in range(n_ts):
    #         x_ts[i] = x_mg[noise_length * (i): noise_length + noise_length * (i)]
    #         # x_ts[i] = (x_ts[i] - x_ts[i].mean()) / x_ts[i].std()
    # elif name == 'narma_10':
    #     total_length = 400000
    #     x_narma_10 = narma(total_length, order=10)
    #     x_ts = np.zeros((n_ts, noise_length))

    #     for i in range(n_ts):
    #         noise = np.random.normal(0, sigma, noise_length)
    #         x_ts[i] = x_narma_10[noise_length * (i): noise_length + noise_length * (i)] + noise


    #         # x_ts[i] = (x_ts[i] - x_ts[i].mean()) / x_ts[i].std()
    #     #     x_ts[i] = x_s[noise_length * (i): noise_length + noise_length * (i)] + noise
    # elif name =='sin':
    #     total_length = 400000
    #     x_s = np.sin(range(0, total_length))
    #     x_ts = np.zeros((n_ts, noise_length))
    #     for i in range(n_ts):
    #         noise = np.random.normal(0, sigma, noise_length)
    #         x_ts[i] = x_s[noise_length * (i): noise_length + noise_length * (i)] + noise
    #         # x_ts[i] = (x_ts[i] - x_ts[i].mean()) / x_ts[i].std()
    #     # standardize

    # Informer datasets
    if 'ett' in name:
        ett_handle = 'h1'
        if 'h2' in name:
            ett_handle = 'h2'
        elif 'm1' in name:
            ett_handle = 'm1'
        elif 'm2' in name:
            ett_handle = 'm2'

        x_mg = np.genfromtxt('data_/ETT-small/ETT' + ett_handle + '.csv', delimiter=',')
        max_length = x_mg.shape[0]

    elif 'ecl' in name:
        d_raw = pd.read_csv('data_/ECL.csv')
        # Assume ts_index contains column names in this case
        x_mg = d_raw['MT_320'].values
        x_mg = x_mg.reshape(x_mg.shape[0], 1)
        max_length = x_mg.shape[0]
        start = 0
        ts_index = np.arange(x_mg.shape[1])
    elif 'weather' in name:
        d_raw = pd.read_csv('data_/WTH.csv')
        # Assume ts_index contains column names in this case
        x_mg = d_raw['WetBulbCelsius'].values
        x_mg = x_mg.reshape(x_mg.shape[0], 1)
        max_length = x_mg.shape[0]
        start = 0
        ts_index = np.arange(x_mg.shape[1])
    # FedFormer datasets:
    elif 'electricity' in name:
        x_mg = np.genfromtxt('data_/electricity/electricity.txt', delimiter=',')
        max_length = x_mg.shape[0]
    elif 'traffic' in name:
        x_mg = np.genfromtxt('data_/traffic/traffic.txt', delimiter=',')
        max_length = x_mg.shape[0]
    elif 'solar' in name:
        x_mg = np.genfromtxt('data_/solar-energy/solar_AL.txt', delimiter=',')
        max_length = x_mg.shape[0]
    elif 'exchange' in name:
        x_mg = np.genfromtxt('data_/exchange_rate/exchange_rate.txt', delimiter=',')
        max_length = x_mg.shape[0]
    elif 'signalz' in name:
        start = 0
        if 'mg' in name:
            x_mg = signalz.mackey_glass(noise_length * n_ts).reshape(noise_length, n_ts)
        elif 'brown' in name:
            x_mg = signalz.brownian_noise(noise_length * n_ts).reshape(noise_length, n_ts)
        elif 'levynoise' in name:
            x_mg = signalz.levy_noise(noise_length * n_ts, alpha=1.5, beta=0.5, sigma=1., position=-2)
            x_mg = x_mg.reshape(noise_length, n_ts)
        elif 'levywalk' in name:
            x_mg = signalz.levy_walk(noise_length * n_ts, ns=500, alpha=1.4, beta=0., sigma=1., position=0)
            x_mg = x_mg.reshape(noise_length, n_ts)
        elif 'logistic' in name:
            x_mg = signalz.signalz.logistic_map(noise_length * n_ts, r=3)
            x_mg = x_mg.reshape(noise_length, n_ts)

    noise_length = min(noise_length, max_length)
    x_ts = np.zeros((noise_length, n_ts))

    if np.isscalar(ts_index):
        ts_index = np.array([ts_index] * n_ts)
    try:
        x_ts[:, :] = x_mg[start: start + noise_length, ts_index]
    except:
        x_ts[:, :] = x_mg[start: start + noise_length]

    return x_ts.T, noise_length

# x_narma_20 = narma(total_length, order = 20, coefficients = [0.3, 0.05, 1.5,0.01])
# x_narma_30 = narma(total_length, order = 30, coefficients = [0.2, 0.004, 1.5,0.201])


def data_split_config_informer(time_series_type, interval_len, test_flag, horizon=None):
    '''
    Determine train/test/val splits as in the informer paper.

    horizon: bogus parameter to keep consistent signature
    '''
    test_size = 0

    if 'ett' in time_series_type:
        if 'h1' in time_series_type:
            train_size = 12 * 30 * 24 - interval_len  # 12 months,
            train_size_conf = 12 * 30 * 24
            validation_size = 4 * 30 * 24  # 4 months
            noise_length = train_size_conf + validation_size + 1
            if test_flag == 1:
                test_size = 4 * 30 * 24  # 4 months
            noise_length = noise_length + test_size
            test_size = 4 * 30 * 24
        elif 'h2' in time_series_type:
            train_size = 12 * 30 * 24 - interval_len  # 12 months,
            train_size_conf = 12 * 30 * 24
            validation_size = 4 * 30 * 24  # 4 months
            noise_length = train_size_conf + validation_size + 1
            if test_flag == 1:
                test_size = 4 * 30 * 24  # 4 months
            noise_length = noise_length + test_size
        elif 'm1' in time_series_type: # * 4 since measured every 15 mins
            train_size = 12 * 30 * 24 * 4 - interval_len  # 12 months,
            train_size_conf = 12 * 30 * 24 * 4
            validation_size = 4 * 30 * 24 * 4  # 4 months
            noise_length = train_size_conf + validation_size + 1
            if test_flag == 1:
                test_size = 4 * 30 * 24 * 4  # 4 months
            noise_length = noise_length + test_size
        elif 'm2' in time_series_type: # * 4 since measured every 15 mins
            train_size = 12 * 30 * 24 * 4 - interval_len  # 12 months,
            train_size_conf = 12 * 30 * 24 * 4
            validation_size = 4 * 30 * 24 * 4  # 4 months
            # train_size = 1000 - interval_len
            # train_size_conf = 1000
            # validation_size = 350
            noise_length = train_size_conf + validation_size + 1
            if test_flag == 1:
                test_size = 4 * 30 * 24 * 4  # 4 months
                # test_size = 350
            noise_length = noise_length + test_size
    elif 'ecl' in time_series_type:
        train_size = 15 * 30 * 24 - interval_len  # 15 months,
        train_size_conf = 15 * 30 * 24
        validation_size = 3 * 30 * 24  # 3 months
        noise_length = train_size_conf + validation_size + 1
        if test_flag == 1:
            test_size = 4 * 30 * 24  # 4 months
        noise_length = noise_length + test_size
    elif 'weather' in time_series_type:
        train_size = 28 * 30 * 24 - interval_len  # 28 months,
        train_size_conf = 28 * 30 * 24
        validation_size = 10 * 30 * 24  # 10 months
        noise_length = train_size_conf + validation_size + 1
        if test_flag == 1:
            test_size = 10 * 30 * 24  # 10 months
        noise_length = noise_length + test_size
    else:
        train_size = 4000
        noise_length = 8000
        test_size = 0

    return train_size, noise_length, test_size, train_size_conf


def data_split_config_ratio(time_series_type, interval_len, test_flag, horizon=1, ratio=[7,1,2]):
    '''
    Determine train/test/val splits with a variable ratio.
    The fedformer paper uses [7, 1, 2].
    The autoformer paper uses [6, 2, 2] for etth, and the rest like fedformer.
    '''
    # test_flag kept so the input is the same as the former function
    test_size = 0

    if 'ett' in time_series_type:
        ett_handle = 'h1'
        if 'h2' in time_series_type:
            ett_handle = 'h2'
        elif 'm1' in time_series_type:
            ett_handle = 'm1'
        elif 'm2' in time_series_type:
            ett_handle = 'm2'
        x_mg = np.genfromtxt('data_/ETT-small/ETT' + ett_handle + '.csv', delimiter=',')
        total_data_num = x_mg.shape[0]
    elif 'ecl' in time_series_type:
        d_raw = pd.read_csv('data_/ECL.csv')
        total_data_num = len(d_raw.index)
    elif 'weather' in time_series_type:
        d_raw = pd.read_csv('data_/WTH.csv')
        total_data_num = len(d_raw.index)
    elif 'electricity' in time_series_type:
        x_mg = np.genfromtxt('data_/electricity/electricity.txt', delimiter=',')
        total_data_num = x_mg.shape[0]
    elif 'traffic' in time_series_type:
        x_mg = np.genfromtxt('data_/traffic/traffic.txt', delimiter=',')
        total_data_num = x_mg.shape[0]
    elif 'solar' in time_series_type:
        x_mg = np.genfromtxt('data_/solar-energy/solar_AL.txt', delimiter=',')
        total_data_num = x_mg.shape[0]
    elif 'exchange' in time_series_type:
        x_mg = np.genfromtxt('data_/exchange_rate/exchange_rate.txt', delimiter=',')
        total_data_num = x_mg.shape[0]
    else:
        total_data_num = 1000

    total_data_num = total_data_num - horizon - interval_len - 1  # leave space for prediction horizon

    train_ratio = ratio[0]; validation_ratio = ratio[1]; test_ratio = ratio[2];
    total_ratio = sum(ratio);

    train_size = int(np.floor(total_data_num * (train_ratio / total_ratio)) - interval_len)
    train_size_conf = int(np.floor(total_data_num * (train_ratio / total_ratio)))

    test_size = int(np.floor(total_data_num * (test_ratio / total_ratio)))
    validation_size = int(np.floor(total_data_num * (validation_ratio / total_ratio)))

    noise_length = train_size_conf + validation_size + test_size

    return train_size, noise_length, test_size, train_size_conf


def data_split_config(*args, horizon=1, ratio=None):
    if ratio is None:
        return data_split_config_informer(*args)
    else:
        return data_split_config_ratio(*args, horizon=horizon, ratio=ratio)

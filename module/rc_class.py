from module._rc_operations import reshape
import numpy as np
from numpy import linalg
import torch.nn as nn
import torch

import time

from module.util import spec_radius
from module._rc_operations import cycle
from module.util import canonical_projection_mtx

# from module.rc_mlp_class import MLP_regression, train_mlp

# if cfg.device != None:
#     device = torch.device(cfg.device)
if torch.cuda.is_available():
    dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")
device = torch.device(dev)
class RC:

    def __init__(self, W, V, h = [], n_original = 0, n_res_threshold = 500, P = []):

        # Structural parameters
        self.W = W
        # self.V = V
        self.h = h
        self.n_res = W.shape[0]
        try:
            self.n_in = V.shape[1]
        except:
            self.nin = 1
        self.spec_rad = np.max(np.abs(linalg.eig(W)[0]))

        # h_flag = 0 if no predefined h and no predefined n_original
        self.h_flag = 0
        # Pre-dilation dimension
        self.n_original = n_original

        self.fit_type = 'ridge'

        # Takes the roll of S in Prop 12 in Complex paper. For the real paper, this is orthogonal so S^\top = S^{-1}
        if isinstance(P, list): # if no input P, that means no change
            self.P = np.eye(self.n_res)
        else:
            self.P = P

        # If P is not provided there's no change, otherwise we do the permutation
        self.V = self.P @ V

        # if cycle we can just roll on:
        self.cycle_flag = False
        # if np.allclose(W, cycle(W.shape[0], spectral_radius = spec_radius(W))):
        #     print('---------- Faster cycle RC process ------------')
        #     self.cycle_flag = True
        # else:
        #     self.cycle_flag = False

        self.upsilon = np.array([[1, 0], [0, -1]])
        self.cycle_upsilon_flag = False
        # Switch this off for now, rolling feels slower..
        # if np.allclose(W, spec_radius(W) * block_diag(*[cycle(W.shape[0] - 2, spectral_radius = 1), self.upsilon])):
        #     print('---------- Faster cycle + upsilon RC process ------------')
        #     self.cycle_upsilon_flag = True
        # else:
        #     self.cycle_upsilon_flag = False

        if (not isinstance(self.h, list)) and self.h.shape[1] != self.n_res:
                self.Jnp = canonical_projection_mtx(self.n_res, self.h.shape[1])
        else:
            self.Jnp = np.eye(self.n_res)

    def train(self, data, train_size, washout, **kwargs):

        W = self.W
        V = self.V
        n_res = self.n_res

        reg = kwargs.get('reg', 1e-9)
        rc_nonlinear = kwargs.get('rc_nonlinear', 0)
        y_pieces = kwargs.get('y_pieces', data[None, washout + 1: washout + train_size + 1])  # WILL BUG if not given


        # RC train
        X = np.zeros((n_res, train_size + washout - washout))
        Yt = np.transpose(y_pieces)  # Yt = Dim , Horizon , # Points

        # print('Yt shape---')
        # print(y_pieces.shape)
        x = np.zeros((n_res, 1))
        u_col = []

        X_collect = []

        # Training RC

        time_state_i = time.time()
        for t in range(train_size + washout):

            u = data[t]  # t^th row

            # if t == 0:
            #     print(x.shape, u.shape, V.shape, W.shape)
            if self.cycle_flag:
                x = np.dot(V, u) + np.roll(x,1)
            elif self.cycle_upsilon_flag:
                x = np.dot(V, u) + spec_radius(W) * np.append(np.roll(x[:-2], 1), self.upsilon @ x[-2:])
            else:
                x = np.dot(V, u) + np.dot(W, x)

            if rc_nonlinear == 1:
                x = np.tanh(x)

            X_collect.append(x)

            if t >= washout:  # washout or beyond we record it
                u_col.append(u)
                X[:, t - washout] = x[:, 0]

        u_col = np.array(u_col)
        time_state_f = time.time()
        # print('State generation time: ' + str(time_state_f - time_state_i))
        # Yt = Yt.squeeze() # RSF: In multivariate case there's an extra dimension

        # Multidimensional ridge by just flattening the target
        # To retrieve the prediction we just reshape it back

        Xt = X.transpose()

        Xt, Y = reshape(Xt, Yt.transpose(), Xt.shape[0])
        X = Xt.transpose()  # n_res,  # points
        Yt = Y.transpose()  # (Dim * Horizon), # points

        # if len(self.h) == 0: # i.e. no input, defaults to []. Then we have to train.
        if isinstance(self.h, list):
            time_regression_i = time.time()
            if self.fit_type == 'ridge':
                # Ridge
                Wout = linalg.solve(np.dot(X, X.T) + reg * np.eye(n_res), np.dot(X, Yt.T)).T
            time_regression_f = time.time()
            # print('Training Wout time: ' + str(time_regression_f - time_regression_i))

            # print('Saving learned Readout.')
            self.h = Wout
        else:
            Wout = self.h
            self.h_flag = 1
            # print('Readout pre-defined.')
        # print(Wout.shape, X.shape, Y.shape)

        return X, Yt, Wout, x, X_collect  # if deep learning after RC just forget about Wout

    def validate(self, data, X_collect, train_size, noise_length, washout, x, **kwargs):

        fit_type = self.fit_type
        n_res = self.n_res
        W = self.W
        V = self.V

        if self.h_flag == 0:
            Wout = self.h
            # print('Self learned Readout.')
        else:
            Wout_fn = ReadOutProjection(self.n_original, self.h, self.P, self.Jnp)
            # print('Loading Readout.')


        rc_nonlinear = kwargs.get('rc_nonlinear', 0)
        horizon = kwargs.get('horizon', 1)
        # RC validate
        val_run_size = kwargs.get('val_run_size', noise_length - train_size - washout - 1)

        val_rec_size = kwargs.get('val_rec_size', int(val_run_size / 2))

        X_val = np.zeros((n_res, val_rec_size))

        Y = np.zeros((horizon, val_rec_size)) # self.h.shape[0] = horizon
        # print(self.h.shape)

        # Predict

        start_pt = kwargs.get('start_pt',
                              train_size + washout)  # we start producing prediction of start_pt +1 using this, in reality val_start should be this +1
        u = data[start_pt]

        u_pred = []

        record_start = kwargs.get('record_start', 0)

        for t in range(val_run_size):

            if self.cycle_flag:
                x = np.dot(V, u) + np.roll(x, 1)
            elif self.cycle_upsilon_flag:
                x = np.dot(V, u) + spec_radius(W) * np.append(np.roll(x[:-2], 1), self.upsilon @ x[-2:])
            else:
                x = np.dot(V, u) + np.dot(W, x)

            if rc_nonlinear == 1:
                x = np.tanh(x)

            if fit_type == 'ridge':
                # print('Ridge validate')
                if self.h_flag == 0:
                    y = np.dot(Wout, x)
                else:
                    y = Wout_fn(x)
            elif fit_type == 'deep':
                # print('Deep validate')
                if self.h_flag == 0:
                    y = Wout(torch.tensor(np.transpose(x).astype(np.float32)).to(device))
                else:
                    y = Wout_fn(x)
                    # y = y.cpu().detach().numpy()
                y = y.cpu().detach().numpy()

            u = data[start_pt + t + 1]

            if t >= record_start:
                try:  # see (+) below
                    X_collect.append(x)
                    X_val[:, t - record_start] = x[:, 0]
                    u_pred.append(u)

                    if horizon == 1:
                        Y[:, t - record_start] = y
                    else:
                        if fit_type == 'deep':
                            Y[:, t - record_start] = y[0]
                        else:
                            Y[:, t - record_start] = y.reshape(-1)
                except:
                    # __import__("pdb").set_trace()
                    continue
            # (+): This is because when rec start = 0, the tail goes on, to avoid this we can either expand X_val and cut the tail or just do try except
        # u_pred.append(u)
        u_pred = np.array(u_pred)

        return X_val, Y, u_pred

    def core(self, data, train_size, washout, noise_length, **conf):


        normalize_flag = conf.get('normalize_flag', 1)
        val_run_size = conf.get('val_run_size', noise_length - washout - train_size - 1)
        record_start = conf.get('record_start', 0)
        ridge_coeff = conf.get('ridge_coeff', 1e-8)
        rc_nonlinear = conf.get('rc_nonlinear', 0)
        max_epochs = conf.get('max_epochs', 100)
        learning_rate = conf.get('learning_rate', 1e-3)
        weight_decay = conf.get('weight_decay', 1e-5)
        batch_size = conf.get('batch_size', 128)

        if self.cycle_flag:
            print('----- Using roll for cyclic -----')
        # print('record_start' + str(record_start))
        # print('val run' + str(val_run_size))

        val_rec_size = conf.get('val_rec_size', int(val_run_size / 2))

        # print('Val REC size' + str(val_rec_size))
        # print(self.h)
        # If already normalized, this step won't change anything.
        # We only need normalized time series for RC, un-normailzed at validation
        # y_pieces already normalized outside
        # target already normalized outside

        # y_val_pieces, target_val NOT normalized!

        if normalize_flag == 1:
            xm = data.mean()
            xs = data.std()
            if np.abs(xm - 0) < 1e-14 and np.abs(xs - 1) < 1e-14:
                normalize_flag = 0
                # print('Signal already normalized...')
            else:
                data = (data - data.mean()) / data.std()
        # Prediction horizon
        horizon = conf.get('horizon', 1)

        time_train_i = time.time()
        # RC train (rc_linear or not)
        y_pieces = conf.get('y_pieces', [])  # WILL BUG if not given properly
        # TODO: add MLP arguments to train IF fit_type == 'deep'
        # Wout in train is ONLY used for output of this function.
        # The Wout we use in validation is stored in self.h
        X, Yt, Wout, x_rc, X_collect = self.train(data, train_size, washout, reg=ridge_coeff, rc_nonlinear=rc_nonlinear,
                                                  horizon=horizon, y_pieces=y_pieces,
                                                  max_epochs = max_epochs, learning_rate = learning_rate,
                                                  batch_size = batch_size, weight_decay = weight_decay)
        time_train_f = time.time()

        # RC native validation
        '''
        Returns:  X_val, val_loss_rc, val_loss_rc_mae, Y
            First get: X_val, Y, u_pred first
            Then Y, u_pred is used to compute val_loss_rc, val_loss_rc_mae
        '''
        time_val_i = time.time()
        X_val, Y, u_pred = self.validate(data, X_collect, train_size, noise_length,
                                         washout, x_rc, rc_nonlinear=rc_nonlinear,
                                         horizon=horizon, record_start=record_start,
                                         val_run_size=val_run_size, val_rec_size=val_rec_size)

        # print(Y.shape)
        criterion = nn.MSELoss()  # mean square error loss
        criterionL1 = nn.L1Loss()
        time_val_f = time.time()

        # print('Validation time: ' + str(time_val_f - time_val_i))

        time_pred_i = time.time()
        val_loss_rc = np.zeros(len(u_pred))
        y_val_pieces = conf.get('y_val_pieces', [])  # Will BUG if we don't give anything!

        if horizon == 1:
            # u_pred = u_pred
            u_pred = y_val_pieces[0]
            val_loss_rc = np.zeros(u_pred.shape[0])
        else:
            u_pred = y_val_pieces[0]
            val_loss_rc = np.zeros(u_pred.shape[0])

        val_loss_rc_mae = np.zeros(u_pred.shape[0])
        # print(Y.shape)
        # print(len(u_pred))

        for i in range(len(u_pred)):
            target = u_pred[i]
            pred = Y[:, i]
            # 29/05/2022: reverse z score for un-normalized signals
            if normalize_flag == 1:
                pred = pred * xs + xm

            val_loss_rc[i] = criterion(torch.tensor(pred), torch.tensor(target))
            val_loss_rc_mae[i] = criterionL1(torch.tensor(pred), torch.tensor(target))


        return X_val, Wout, val_loss_rc, val_loss_rc_mae, Y


class ReadOutProjection():

    def __init__(self, n_original, Wout, P, Jnp):
        '''
        Assume Wout is the big one after dilation
        n_original is the pre-dilation dimension, if same as current state space dimension n_res
        then it works just as before
        '''
        self.n_original = n_original
        self.Wout = Wout
        self.P = P
        self.Jnp = Jnp

    def __call__(self, state):
        '''
        State -- reservoir state, dilated form
        '''

        if state.reshape(-1).shape[0] == self.Wout.shape[1]:
            # Meaning we don't' need to do h(pullback), original form
            # print('original')
            # return np.dot(self.Wout, np.transpose(self.P) @ state)
            return np.dot(self.Wout, np.transpose(self.Jnp) @ np.transpose(self.P) @ state)
        else:
            # pullback to the first n_res entries of output,
            # i.e. discarded Wout.shape[1] - n_res > 0 number of entries at the end.
            # print('dilation')
            # return np.dot(self.Wout, (np.transpose(self.P) @  state)[:self.n_original])
            # print(state.shape)
            return self.Wout @ (np.transpose(self.Jnp) @ np.transpose(self.P) @  state)[:self.n_original]

        # FIXME: if self.n_res >= n_res_threshold then we do MLP





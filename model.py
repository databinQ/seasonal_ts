# coding: utf-8

from utils.ts import *
from utils.common import *


class SARIMAXModel(object):
    def __init__(self, n_dev=12, n_test=24, use_exog=False):
        self.n_dev = n_dev
        self.n_test = n_test
        self.use_exog = use_exog
        self.model = None

    def train(self, ts, exogs=None, method="lowess", ma_method="sma", window=3, decay_factor=2, span=3):
        ts_array = check_array(ts)

        n_ts = len(ts)
        if n_ts <= 2 * self.n_dev:
            raise ValueError("Time series is too short, must be longer than {}, got {}".format(2 * self.n_dev, n_ts))

        # check additive pattern
        if not check_add_pattern(ts_array):
            # use `log` transform function, change 0 to 1
            ts_array[ts_array == 0] = 1
            ts_new = np.log(ts_array)
        else:
            ts_new = ts_array

        n_train = n_ts - self.n_dev
        train_ts = ts_new[:-self.n_dev]
        dev_ts = ts_new[-self.n_dev:]

        # handle exogenous variable
        if self.use_exog:
            if exogs is None:
                '''use time index as exog while none exogenous variable input'''
                train_time_index = np.arange(1, n_train + 1)
                dev_time_index = np.arange(n_train + 1, n_ts + 1)
                test_time_index = np.arange(n_ts + 1, n_ts + self.n_test + 1)
                dev_test_time_index = np.hstack([dev_time_index, test_time_index])

                train_exog, dev_test_exog = find_exogs(
                    train_time_index, train_ts, dev_test_time_index, method=method, ma_method=ma_method, window=window,
                    decay_factor=decay_factor, span=span
                )

            else:
                if exogs.ndim < 2:
                    exogs = exogs.reshape((-1, 1))

                # use moving average to predict features' value
                n_feature = exogs.shape[1]
                dev_test_feature = np.zeros(shape=(self.n_dev + self.n_test, n_feature))
                for k in range(n_feature):
                    _, t = ewma(train_ts, fcst_period=dev_test_feature.shape[0], span=span)
                    dev_test_feature[:, k] = t

                train_exog, dev_test_exog = find_exogs(
                    exogs, train_ts, dev_test_feature, method=method, ma_method=ma_method, window=window,
                    decay_factor=decay_factor, span=span
                )

            train_x = train_ts - train_exog
            dev_x = dev_ts - dev_test_exog[:self.n_dev]
        else:
            train_x = train_ts
            dev_x = dev_ts

        # find difference steps
        d = find_diff(train_x)
        ts_diff = np.diff(train_x, d)

        # find frequency of seasonal
        frequency = seasonal_detection(ts_diff)

        # find seasonal difference steps
        if frequency == 0:
            D = 0
        else:
            D = cal_seasonal_diff(ts_diff, frequency)
        if D == 0:
            ts_seasonal_diff = ts_diff
        else:
            ts_seasonal_diff = seasonal_diff(ts_diff, D, frequency)

        p, q, P, Q = SARIMA_search(ts_seasonal_diff, frequency, D=D)
        try:
            if P + D + Q == 0:
                if max(p + P, q + Q) == 0:
                    q = 1
                if self.use_exog:
                    model = SARIMAX(train_ts, order=(p, d, q), exog=train_exog, trend="c")
                else:
                    model = SARIMAX(train_ts, order=(p, d, q), trend="c")
            else:
                if max(p + P, q + Q) == 0:
                    Q = 1
                if self.use_exog:
                    model = SARIMAX(train_ts, order=(p, d, q), seasonal_order=(P, D, Q, frequency), exog=train_exog,
                                    trend="c")
                else:
                    model = SARIMAX(train_ts, order=(p, d, q), seasonal_order=(P, D, Q, frequency), trend="c")
        except Exception as e:
            print("Fitting error")
            print(type(e), e)

        fit_results = model.fit(method="lbfgs", disp=False)
        fitted = fit_results.fittedvalues
        if self.use_exog:
            pred = fit_results.forecast(steps=self.n_dev + self.n_test, exog=dev_test_exog[:, None])
        else:
            pred = fit_results.forecast(steps=self.n_dev + self.n_test)

        fitted = np.exp(fitted)
        pred = np.exp(pred)
        dev_pred, test_pred = pred[:self.n_dev], pred[self.n_dev:]
        rmse = RMSE(ts_array[-self.n_dev:], dev_pred)
        return fitted, pred, rmse

# coding: utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, RANSACRegressor

SEED = 2019


def check_array(ts):
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)
    if ts.ndim > 1:
        return ts.ravel()
    return ts.copy()


def shift_left(array, step=1):
    res_array = array.copy()
    res_array[:-step] = res_array[step:]
    res_array[-step:] = 0
    return res_array


def shift_right(array, step=1):
    if step == 0:
        return array.copy()
    res_array = array.copy()
    res_array[step:] = res_array[:-step]
    res_array[:step] = 0
    return res_array


def sma(ts, fcst_period, window=3, decay_factor=2):
    x_fit = np.zeros(len(ts))
    x_fit[:(window-1)] = ts[:(window-1)]
    x_pred = np.zeros(fcst_period)

    weights = np.array([decay_factor ** i for i in range(window)])
    weights = weights / weights.sum()

    x_fit[(window-1):] = np.convolve(ts, weights, mode="valid")

    tmp_ts = x_fit[-window:]
    for i in range(fcst_period):
        x_pred[i] = np.matmul(tmp_ts, weights)
        tmp_ts = shift_left(tmp_ts)
        tmp_ts[-1] = x_pred[i]
    return x_fit, x_pred


def ewma(ts, fcst_period, span=3):
    if isinstance(ts, np.ndarray):
        ts = pd.Series(ts)
    x_fit = ts.ewm(span=span).mean()
    x_pred = np.zeros(fcst_period)

    tmp_ts = ts.iloc[-(span+1):].copy()
    tmp_ts.iloc[-1] = x_fit.iloc[-1]

    for i in np.arange(fcst_period):
        x_pred[i] = tmp_ts.ewm(span=span).mean().iloc[-1]
        tmp_ts = shift_left(tmp_ts)
        tmp_ts[-1] = x_pred[i]
        tmp_ts = pd.Series(tmp_ts)
    return x_fit.values, x_pred


def simple_lr_predict(x, y, tx):
    model = LinearRegression()
    model.fit(x, y)
    return model.predict(tx[:None])


def lowess_fit_predict(x, y, tx, frac=0.3):
    lowess_res = sm.nonparametric.lowess(endog=y, exog=x, frac=frac)
    lowess_y = lowess_res[:, -1]
    preds = simple_lr_predict(x[:, None], lowess_y, tx[:, None])
    return lowess_y, preds


def ransac_fit_predict(x, y, tx, iters=100):
    """
    :param iters: total number of individual `RANSACRegressor` model. Return meaning of the prediction of all models.
    """
    pred_res = []
    for _ in range(iters):
        model_ransac = RANSACRegressor(LinearRegression(), max_trials=10000, random_state=SEED)
        model_ransac.fit(x[:, None], y)
        pred_res.append(model_ransac.predict(tx[:, None]))
    return np.mean(pred_res, axis=0)

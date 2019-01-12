# coding: utf-8

import sys
import itertools
import numpy as np
from functools import partial
from scipy.stats import norm
from numpy.linalg import lstsq, svd
from statsmodels.tsa.stattools import acf, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.common import *

methods = ("lowess", "robust", "movave")
ma_methods = ("sma", "ewma")

MAX_P, MAX_D, MAX_Q = 5, 3, 5
MAX_PP, MAX_DD, MAX_QQ = 2, 2, 2


def check_add_pattern(ts):
    add_results = seasonal_decompose(ts, model="additive", freq=12)
    mul_results = seasonal_decompose(ts, model="multipliative", freq=12)
    add_resid = add_results.resid
    add_resid = add_resid[~np.isnan(add_resid)]
    mul_resid = mul_results.resid
    mul_resid = mul_resid[~np.isnan(mul_resid)]
    add_acf_total = np.sum(np.square(acf(add_resid)))
    mul_acf_total = np.sum(np.square(acf(mul_resid)))
    return add_acf_total < mul_acf_total


def find_exogs(x, y, tx, method="lowess", ma_method="sma", window=3, decay_factor=2, span=3):
    if method not in methods:
        raise ValueError("method must be one of {}, got {} instead".format(methods, method))

    try:
        if method == "lowess":
            return lowess_fit_predict(x, y, tx)
        elif method == "robust":
            t_tx = np.hstack([x, tx])
            pred_y = ransac_fit_predict(x, y, t_tx)
            return pred_y[:len(x)], pred_y[len(x):]
        elif method == "movave":
            if ma_method not in ma_methods:
                raise ValueError("moving average method must be one of {}, got {} instead".format(ma_methods,
                                                                                                  ma_method))
            if ma_method == "sma":
                return sma(y, fcst_period=len(tx), window=window, decay_factor=decay_factor)
            elif ma_method == "ewma":
                return ewma(y, fcst_period=len(tx), span=span)
    except Exception:
        print("warning: using simple linear regression model to get exogs")
        t_tx = np.hstack([x, tx])
        pred_y = simple_lr_predict(x, y, t_tx)
        return pred_y[:len(x)], pred_y[len(x):]


def stationary_test(ts, kpss_reg_type="c", significance=5):
    try:
        result = kpss(ts, regression=kpss_reg_type)
        p_value = result[1]
        return True if p_value >= significance / 100 else False
    except Exception:
        return False


def find_diff(ts):
    """If time series is still not stationary after `MAX_D`, return `MAX_D`"""
    d = 0
    for d in range(MAX_D + 1):
        tmp_ts = np.diff(ts, d)
        if stationary_test(tmp_ts):
            break
    return d


def fft_test(ts, freq):
    fix_len = len(ts) // freq * freq
    tmp_ts = ts[:fix_len]
    fft_res = np.fft.fft(tmp_ts)
    p = np.power(np.abs(fft_res), 2)  # `abs`的作用是对结果中的虚数取模
    tmp_p = p[1:(fix_len // 2)]  # 截半的原因是如果频率超过了一半, 就会有部分位置没有上个(或下个)周期对应的点, 就没有了参考意义
    max_power = np.max(tmp_p)
    max_index = 1 + np.argmax(tmp_p)

    try:
        return freq % (fix_len // max_index) == 0
    except ZeroDivisionError:
        return False


def acf_test(ts, freq, significance=5):
    try:
        norm_clim = norm.ppf((2 - significance / 100) / 2) / np.sqrt(len(ts))
        acf_res = acf(ts, unbiased=True, nlags=freq)
        return acf_res[freq] >= norm_clim
    except (TypeError, IndexError):
        return False


def seasonal_detection(ts, recommend_freqs=(3, 6, 12), sigificance=5):
    for frequency in recommend_freqs:
        fft_test_res = fft_test(ts, frequency)
        acf_test_res = acf_test(ts, frequency, sigificance)
        if fft_test_res and acf_test_res:
            return frequency
    return 0


def seasonal_dummy(ts, frequency):
    """generate seasonal dummy matrix using Fourier series for Canova-Hansen test."""
    n, m = len(ts), frequency
    tt = np.arange(1, n + 1, 1)
    mat = np.zeros([n, 2 * m], dtype=float)
    for i in np.arange(0, m):
        mat[:, 2*i] = np.cos(2.0 * np.pi * (i + 1) * tt / m)
        mat[:, 2*i+1] = np.sin(2.0 * np.pi * (i + 1) * tt / m)
    return mat[:, 0:(m-1)]


def sd_statstics(full_ts, frequency):
    if frequency <= 1:
        return 0
    N = len(full_ts)
    if N <= frequency:
        return 0

    frec = np.ones((frequency + 1) // 2, dtype=np.int32)
    l_trunc = int(np.round(np.power(N / 100, 0.25) * frequency))
    r1 = seasonal_dummy(full_ts, frequency)
    r1wInterceptCol = np.column_stack([np.ones(r1.shape[0], dtype=float), r1])
    lstsq_result = lstsq(a=r1wInterceptCol, b=full_ts)
    residual = full_ts - np.matmul(r1wInterceptCol, lstsq_result[0])
    f_hat = np.zeros([N, frequency - 1], dtype=float)
    f_hat_aux = np.zeros([N, frequency - 1], dtype=float)
    for i in np.arange(0, frequency - 1):
        f_hat_aux[:, i] = r1[:, i] * residual
    for i in np.arange(0, N):
        for j in np.arange(0, frequency - 1):
            my_sum = sum(f_hat_aux[0:(i+1), j])
            f_hat[i,j] = my_sum

    wnw = np.ones(l_trunc, dtype=float) - np.arange(1, l_trunc + 1, 1) / (l_trunc + 1)
    Ne = f_hat_aux.shape[0]
    omnw = np.zeros([f_hat_aux.shape[1], f_hat_aux.shape[1]], dtype=float)
    for k in range(0, l_trunc):
        omnw = omnw + np.matmul(f_hat_aux.T[:, (k+1):Ne], f_hat_aux[0:(Ne-(k+1)), :]) * float(wnw[k])

    cross = np.matmul(f_hat_aux.T, f_hat_aux)
    omnw_plus_transpose = omnw + omnw.T
    omf_hat = (cross + omnw_plus_transpose) / float(Ne)

    sq = np.arange(0, frequency - 1, 2)
    frecob = np.zeros(frequency - 1, dtype=int)
    for i in np.arange(0, len(frec)):
        if (frec[i] == 1) & ((i + 1) == int(frequency / 2.0)):
            frecob[sq[i]] = 1
        if (frec[i] == 1) & ((i + 1) < int(frequency / 2.0)):
            frecob[sq[i]] = 1
            frecob[sq[i]+1] = 1

    a = frecob.tolist().count(1)
    A = np.zeros([frequency - 1, a], dtype=float)
    j = 0
    for i in np.arange(0, frequency - 1):
        if frecob[i] == 1:
            A[i, j] = 1
            j += 1

    aTomfhat = np.matmul(A.T, omf_hat)
    tmp = np.matmul(aTomfhat, A)
    machineDoubleEps = 2.220446e-16
    problems = min(svd(tmp)[1]) < machineDoubleEps
    if problems:
        stL = 0.0
    else:
        solved = np.linalg.solve(tmp, np.eye(tmp.shape[1], dtype=float))
        step1 = np.matmul(solved, A.T)
        step2 = np.matmul(step1, f_hat.T)
        step3 = np.matmul(step2, f_hat)
        step4 = np.matmul(step3, A)
        stL = (1.0 / np.power(N, 2.0)) * sum(np.diag(step4))
    return stL


def ch_test(full_ts, frequency):
    if len(full_ts) < 2 * frequency + 5:
        return False
    crit_values = {2: 0.353, 3: 0.610, 4: 0.846, 5: 1.070, 6: 1.280, 7: 1.490, 8: 1.690, 9: 1.890, 10: 2.100, 11: 2.290,
                   12: 2.490, 13: 2.690, 24: 5.098624, 52: 10.341416, 365: 65.44445}
    ch_stat = sd_statstics(full_ts, frequency)
    if frequency not in crit_values:
        return ch_stat <= 0.269 * np.power(frequency, 0.928)
    return ch_stat <= crit_values[frequency]


def seasonal_diff(ts, order=1, frequency=12, padding=False):
    n = len(ts)
    ts_padding_array = np.array([])

    if order == 0:
        return ts
    if n <= frequency * order:
        return ts

    tmp1 = ts.copy()
    for _ in range(order):
        tmp_diff = tmp1[frequency:] - tmp1[:-frequency]
        if padding:
            ts_padding_array = np.hstack([ts_padding_array, tmp1[:frequency]])
        tmp1 = tmp_diff.copy()
    if padding:
        tmp_diff = np.hstack([ts_padding_array, tmp_diff])
    return tmp_diff


def cal_seasonal_diff(ts, frequency):
    D = 0
    for D in range(0, MAX_DD + 1):
        if D != 0:
            ts = seasonal_diff(ts, D, frequency=frequency)
        if ch_test(ts, frequency):
            return D
    return D


def SARIMA_evaluate(ts, params, freq):
    if params[0] > MAX_P | params[1] > MAX_Q | params[2] > MAX_PP | params[3] > MAX_QQ:
        return params + [sys.maxsize]

    if params[2] + params[3] == 0:
        freq = 0

    try:
        model = SARIMAX(ts, order=(params[0], 0, params[1]), seasonal_order=(params[2], 0, params[3], freq), trend="c")
        model_fit = model.fit(disp=False)
        aic = model_fit.aic
    except ValueError:
        aic = sys.maxsize
    return params + [aic]


def SARIMA_search(ts, frequency, D):
    cur_best_aic = sys.maxsize
    cur_best_params = []
    past_params = set()
    b_finished = False

    next_params = [[2, 2, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]] if frequency != 0 else \
        [[2, 2, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]

    pd_deltas = [-1, 0, 1]
    PQ_deltas = [-1, 0, 1] if frequency != 0 else [0, 0, 0]

    while not b_finished:
        model_aic_list = list(map(partial(SARIMA_evaluate, ts, freq=frequency), next_params))
        new_best_aic = min(t[4] for t in model_aic_list)
        if new_best_aic == sys.maxsize:
            b_finished = False
        if new_best_aic < cur_best_aic:
            cur_best_aic = new_best_aic
            for t in model_aic_list:
                if t[4] == new_best_aic:
                    cur_best_params = t[:4]

        for params in next_params:
            past_params.add(tuple(params))

        next_params = []
        for params_delta in itertools.product(pd_deltas, pd_deltas, PQ_deltas, PQ_deltas):
            p = max(0, cur_best_params[0] + params_delta[0])
            q = max(0, cur_best_params[1] + params_delta[1])
            P = max(0, cur_best_params[2] + params_delta[2])
            Q = max(0, cur_best_params[3] + params_delta[3])
            next_params.append([p, q, P, Q])

        next_params = [params for params, _ in itertools.groupby(next_params) if tuple(params) not in past_params]
        if len(next_params) == 0:
            b_finished = True
        if cur_best_params[2] + cur_best_params[3] + D == 0:
            frequency = 0
    return cur_best_params


def RMSE(data, pred_data):
    return np.sqrt(np.sum(np.power(data - pred_data, 2)) / len(data))

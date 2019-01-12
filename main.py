# coding: utf-8

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import SARIMAXModel

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    data = pd.read_csv("data/AirPassengers.csv", index_col="Month",
                       date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m"))

    model = SARIMAXModel(use_exog=True)
    train_fitted, dev_test_pred, errors = model.train(data)

    plt.plot(data.values, "g-", label="raw")
    plt.plot(np.hstack([train_fitted, dev_test_pred]), color="red", label="model")
    plt.legend()
    plt.show()

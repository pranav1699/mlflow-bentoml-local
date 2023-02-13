import os
import warnings
import sys
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
from PIL import Image




diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = diabetes.feature_names + ["progression"]
data = pd.DataFrame(d, columns=cols)

print(data)

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    train, test = train_test_split(data)

    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")

    eps = 5e-3  # the smaller it is the longer is the path

    print("Computing regularization path using the elastic net.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio)

    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle="--", c=c)

    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    title = "ElasticNet Path by alpha for l1_ratio = " + str(l1_ratio)
    plt.title(title)
    plt.axis("tight")

    fig.savefig("ElasticNet-paths.png")
    plt.close(fig)
    

    mlflow.log_artifact("ElasticNet-paths.png")
    

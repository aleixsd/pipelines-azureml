from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import azureml.core
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.environment import Environment
import pandas as pd
import os

from azureml.core.run import Run

import sklearn

run = Run.get_context()

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}
		

alpha = 0.5
run.log("alpha_value", alpha)

reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])

preds = reg.predict(data["test"]["X"])

mse = mean_squared_error(preds, y_test)
run.log("mse", mse)

model_name = "sklearn_regression_model.pkl"

joblib.dump(value=reg, filename="./outputs/" + model_name)

#run.upload_file(name=model_name, path_or_stream="outputs/" + model_name)

import os
import sys

import numpy as np
import pandas as pd
import dill 
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    try:
        report = {}

        for model_name, model in models.items():
 
            model_parameters = parameters.get(model_name, {})

            if model_parameters:

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=model_parameters,
                    n_iter=9,
                    cv=3,
                    scoring="r2",
                    random_state=42,
                    n_jobs=-1
                )

                random_search.fit(X_train, y_train)

                best_model = random_search.best_estimator_

            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path: str) -> object: 
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
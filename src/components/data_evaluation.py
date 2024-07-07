import os
import sys
import numpy as np
from src.utils.utils import load_object
from src.logger.logging import logging
from src.exception.exception import customexception
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self, actual, pred):
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        return mse, rmse, mae, r2
    
    def initiated_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])
            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)

            pred = model.predict(X_test)

            mse, rmse, mae, r2 = self.eval_metrics(y_test, pred)

            information = (f"Mean Square Error(MAE) : {mse}\n Root Mean Square Error: {rmse}\n Mean Absolute Error (MAE) : {mae} \n r2 Score : {r2}")

            logging.info(information)
            print(information)

        except Exception as e:
            raise customexception(e,sys)
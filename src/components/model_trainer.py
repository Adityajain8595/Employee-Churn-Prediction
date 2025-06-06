import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
            "Decision Tree Classifier":DecisionTreeClassifier(),
            "Logistic Regression":LogisticRegression(),
            'K-Nearest Neighbours Classifier':KNeighborsClassifier(),
            "Random Forest Classifier":RandomForestClassifier(),
            'AdaBoost Classifier':AdaBoostClassifier(),
            'Gradient Boosting Classifier': GradientBoostingClassifier(),
            'XgBoost Classifier':XGBClassifier()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            logging.info("Model Report: {}".format(model_report))

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score[0] < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            f1 = f1_score(y_test,predicted)

            return (
                best_model_name,
                f1
            )
        
        except Exception as e:
            raise CustomException(e,sys)
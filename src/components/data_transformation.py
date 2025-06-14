import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        
        try:
            
            numerical_columns =  ['satisfaction_level', 'last_evaluation', 'number_project','average_montly_hours', 
                                   'time_spend_company', 'Work_accident','promotion_last_5years']
            categorical_columns = ['Department', 'salary']

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),
                    ("OneHotEncoder",OneHotEncoder(drop='first'))
                ]
            )
            logging.info("Categorical columns one-hot encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical Pipeline", num_pipeline, numerical_columns),
                    ("Categorical Pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except CustomException as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test datasets completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'left'

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_features_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_features_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            sm = SMOTE(random_state=42)
            input_feature_train_arr, target_features_train_df = sm.fit_resample(
                input_feature_train_arr, target_features_train_df
            )

            logging.info("Applied SMOTE to balance the training dataset")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_features_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_features_test_df)
            ]
            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)
            
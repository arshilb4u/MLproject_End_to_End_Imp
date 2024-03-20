import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logger
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def data_transformation_object(self):

        '''
        this function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numericalpipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= 'median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_Pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= 'most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean= False))
                    ]
                 )
            logger.info(f"Categorical column :{categorical_columns}")

            logger.info(f"Numerical column :{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipelines", numericalpipeline,numerical_columns),
                    ("cat_pipeline", cat_Pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Read train and test data completed")

            logger.info("getting preprocessing object")
            preprocessor_obj = self.data_transformation_object()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logger.info("Applying preprocessing object on training dataframe and testing dataframe.")

            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]

            logger.info(f"Saved preprocessing object.")
            
            save_object(

                file_path = self.config.preprocessor_obj_file_path,
                obj = preprocessor_obj

            )

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
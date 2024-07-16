import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler 

from source.exception import CustomException
from source.logger import logging 
from source.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):

        # """
        # This function is responsible for data transformation based on different types of data.

        # """

        try:

            num_features = ["writing_score","reading_score"]
            categorical_features = [
                "gender" , "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputing" , SimpleImputer(strategy = "median")),
                    ("scaling",StandardScaler(with_mean=False))

                ]
               
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputing",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoding" , OneHotEncoder()),
                    ("sclaing",StandardScaler(with_mean=False))
                    
                    
                ]

                
            )
            logging.info(f"Numerical Features:{num_features}")
            logging.info(f"Categorical Features:{categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipleline", num_pipeline , num_features),
                    ("cat_pipeline", cat_pipeline , categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path , test_path):

        try :

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test dataframes created.")

            logging.info("creatin preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_col = 'math_score'

            numerical_cols = ['writing_score','reading_score']

            input_feature_train = train_df.drop(columns = [target_col],axis=1)
            target_feature_train = train_df[target_col]

            input_feature_test = test_df.drop(columns = [target_col],axis=1)
            target_feature_test = test_df[target_col]


            logging.info("Applying preprocessing object on the train and test dataframes.")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train)

            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test)

            ]

            logging.info("saved preprocessing object.")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path , 
                        obj = preprocessing_obj)




            return (
                train_arr,
                test_arr ,
                self.data_transformation_config.preprocessor_obj_file_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        
        































       




    
        










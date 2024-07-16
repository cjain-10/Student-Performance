import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,
GradientBoostingRegressor,RandomForestRegressor)

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from source.exception import CustomException
from source.logger import logging


from source.utils import save_object , evaluate_models


@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):

        logging.info("starting model training.")

        try:
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {"Linear Regression": LinearRegression(),
                      
                      "DecisionTreeRegressor":DecisionTreeRegressor(),
                      "XGBRegressor":XGBRegressor(),"CatBoostRegressor":CatBoostRegressor(),
                      "AdaBoostRegressor":AdaBoostRegressor() , 
                      "GradientBoostingRegressor":GradientBoostingRegressor(),
                      "RandomForestRegressor":RandomForestRegressor()

                      }
            
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }


            
            report:dict = evaluate_models(X_train , y_train,X_test,y_test,models,params)

           

            best_model_score = max(sorted(report.values()))

            best_model_name = list(report.keys())[list(report.values()).index(best_model_score)]

            best_model_dict = {best_model_name : best_model_score}

        


            if best_model_score < 0.6 :
                print("No good model found. Try tuning the hyperparameters.")

            logging.info("found the best model for training and testing data.")

            print(f"The best model on the given data is:{best_model_name} and the model r2 score is:{best_model_score}")
            save_object(self.model_trainer_config.trained_model_file_path, report )


        except Exception as e:
            raise CustomException(e,sys)
    


                

            
            






    


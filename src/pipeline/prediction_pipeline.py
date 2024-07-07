import os
import sys
import pandas as pd
from src.exception.exception import customexception
from src.logger.logging import logging
from src.utils.utils import load_object


class PredictPipeline:

    
    def __init__(self):
        print("init.. the object")

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            scaled_fea=preprocessor.transform(features)
            pred=model.predict(scaled_fea)

            return pred

        except Exception as e:
            raise customexception(e,sys)


class CustomData:
    def __init__(self,
                 venue:str,
                 bat_team:str,
                 bowl_team:str,
                 batsman:str,
                 bowler:str,
                 runs:int,
                 wickets:int,
                 overs:float,
                 runs_last_5:int,
                 wickets_last_5:int):
        
        self.venue=venue
        self.bat_team=bat_team
        self.bowl_team=bowl_team
        self.batsman=batsman
        self.bowler=bowler
        self.runs=runs
        self.wickets = wickets
        self.overs = overs
        self.runs_last_5 = runs_last_5
        self.wickets_last_5=wickets_last_5
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'venue':[self.venue],
                'bat_team':[self.bat_team],
                'bowl_team':[self.bowl_team],
                'batsman':[self.batsman],
                'bowler':[self.bowler],
                'runs':[self.runs],
                'wickets':[self.wickets],
                'overs':[self.overs],
                'runs_last_5':[self.runs_last_5],
                'wickets_last_5':[self.wickets_last_5]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)
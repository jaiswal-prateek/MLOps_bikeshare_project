import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bike_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import extract_year_month
from bikeshare_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bike_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    # print('validated_data', type(validated_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    # print('validated_data after reindex', type(validated_data))

    # print(validated_data.columns)
    results = {"predictions": None, "version": _version, "errors": errors}
    # print('initial results', results['predictions'])
    predictions = bike_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    # print(results)

    if not errors:

        predictions = bike_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday':['2012-02-09'],'season':['spring'],'hr':["11am"], 'holiday':['No'],'weekday':['Thu'],
                'workingday':['Yes'],'weathersit':['Clear'],'temp':[3.28],'atemp':[-0.9982],
                'hum':[52.0],'windspeed':[15.0013], 'casual': [4], 'registered': [95]}
    
    make_prediction(input_data=data_in)

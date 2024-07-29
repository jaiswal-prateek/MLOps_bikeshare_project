import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder
from bikeshare_model.processing.features import ColumnDropper

bike_pipe = Pipeline([

    # Imputation steps
    ('weekday_imputer', WeekdayImputer(config.model_config.weekday_var)),
    ('weathersit_imputer', WeathersitImputer('weathersit')),

    # Mapping steps
    ('year_mapper', Mapper('year', mappings={2011: 0, 2012: 1})),
    ('month_mapper', Mapper('month', mappings={'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7,
                                               'September':8, 'October':9, 'November':10, 'December':11})),
    ('season_mapper', Mapper('season', mappings= {'winter': 0, 'fall': 1, 'spring': 2, 'summer': 3})),
    ('weathersit_mapper', Mapper('weathersit', mappings= {'Clear': 0, 'Mist': 1, 'Light Rain': 2, 'Heavy Rain': 3})),
    ('holiday_mapper', Mapper('holiday', mappings= {'No': 0, 'Yes': 1})),
    ('workingday_mapper', Mapper('workingday', mappings= {'No': 0, 'Yes': 1})),
    ('hr_mapper', Mapper('hr', mappings= {'12am': 0, '1am': 1, '2am': 2, '3am': 3, '4am': 4, '5am': 5, '6am': 6, '7am': 7, '8am': 8, '9am': 9, '10am': 10, '11am': 11, '12pm': 12,
                                          '1pm': 13, '2pm': 14, '3pm': 15, '4pm': 16, '5pm': 17, '6pm': 18, '7pm': 19, '8pm': 20, '9pm': 21, '10pm': 22, '11pm': 23})),

    # Handling Outlier steps
    ('temp_handler', OutlierHandler('temp')),
    ('atemp_handler', OutlierHandler('atemp')),
    ('humidity_handler', OutlierHandler('hum')),
    ('windspeed_handler', OutlierHandler('windspeed')),

    # Weekday Onehot encoding
    ('weekday_onehot', WeekdayOneHotEncoder('weekday')),

    # Column Dropper
    # ('column_dropper', ColumnDropper(cols=['casual', 'registered', 'weekday'])),

    # scale
    ('scaler', StandardScaler()),

    # model
    ('model_rfr', RandomForestRegressor(n_estimators= config.model_config.n_estimators, 
                                        max_depth= config.model_config.max_depth,
                                        random_state= config.model_config.random_state))
    ])
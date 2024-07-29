
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer


def test_weekday_variable_transformer(sample_input_data):

    sample_input_data[0]['dteday'] = pd.to_datetime(sample_input_data[0]['dteday'], format= '%Y-%m-%d')

    # Given
    transformer = WeekdayImputer(
        variable=config.model_config.weekday_var,  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[5,'weekday'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5,'weekday'] == 'Sun'


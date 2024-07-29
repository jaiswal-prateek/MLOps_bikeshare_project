# """
# Note: These tests will fail if you have not first trained the model.
# """
# import sys
# from pathlib import Path
# file = Path(__file__).resolve()
# parent, root = file.parent, file.parents[1]
# sys.path.append(str(root))

# import numpy as np
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# from bikeshare_model.predict import make_prediction


# def test_make_prediction(sample_input_data):
#     # Given
#     # expected_no_predictions = 179
    
#     # When
#     result = make_prediction(input_data=sample_input_data[0])
#     assert result.get("errors") is None  # Assert no errors occurred

#     # Then
#     predictions = result.get("predictions")
#     # assert isinstance(predictions, np.ndarray)
#     # assert isinstance(predictions[0], np.float32)
#     # assert result.get("errors") is None
#     # # assert len(predictions) == expected_no_predictions
#     _predictions = list(predictions)
#     y_true = sample_input_data[1]
#     r2_score_var = r2_score(_predictions, y_true)
#     assert r2_score_var > 0.8


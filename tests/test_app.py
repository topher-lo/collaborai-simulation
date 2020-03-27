import pandas as pd

from pandas.testing import assert_frame_equal

from src.simulation import ml_loader


def test_ml_loader():
    """Test assumes weights_3_3.csv (3 players, 3 time periods) exists. 
    Function converts and returns weights.csv as a dataframe.
    """
    result = pd.read_csv('logs/weights_3_3.csv')
    assert_frame_equal(ml_loader(None, None, 3, 3), result)
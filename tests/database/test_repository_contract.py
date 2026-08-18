import pytest
from src.database.repository import upsert_prediction
def test_prediction_probability_validation():
    with pytest.raises(ValueError): upsert_prediction(None,None,None,'S0',1.1,.5)

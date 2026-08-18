import pytest
from src.database.repository import upsert_prediction
@pytest.mark.parametrize('p,t',[(-.1,.5),(.5,1.1)])
def test_probability_bounds(p,t):
    with pytest.raises(ValueError): upsert_prediction(None,None,None,'S0',p,t)

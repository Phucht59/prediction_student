from src.database.repository import _json
def test_json_values_are_serializable():
    assert isinstance(_json({'a':1}),str)

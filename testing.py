from backend import app
from fastapi.testclient import TestClient

client  = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "AI Doctor Backend Running!"}
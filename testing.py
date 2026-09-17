import mongomock
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# ── Patch MongoDB before importing your app ──────────────────────────────────
@pytest.fixture(autouse=True, scope="session")
def mock_mongo():
    """Replace MongoClient with an in-memory mongomock client."""
    with patch("backend.MongoClient", return_value=mongomock.MongoClient()): # this make sure mongo client is replaced with mongomonk
        yield

from backend import app  # import AFTER patching

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────────────
SAMPLE_PATIENT = {
    "name": "John Doe",
    "age": 30,
    "gender": "Male",
    "blood_group": "O+",
    "contact": "9999999999",
    "medical_history": "None",
    "reports": []
}

def create_patient() -> dict:
    """Utility: create a patient and return the response JSON."""
    res = client.post("/patients", json=SAMPLE_PATIENT)
    assert res.status_code == 201
    return res.json()["patient"]


# ── Root ─────────────────────────────────────────────────────────────────────
def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "AI Doctor Backend Running!"}


# ── CREATE ───────────────────────────────────────────────────────────────────
def test_create_patient():
    res = client.post("/patients", json=SAMPLE_PATIENT)
    assert res.status_code == 201

    data = res.json()["patient"]
    assert data["name"] == "John Doe"
    assert data["uid"].startswith("PAT-")      # uid format check
    assert "created_at" in data


def test_create_patient_with_report():
    payload = {**SAMPLE_PATIENT, "reports": [{
        "report_type": "Blood Test",
        "file_name": "blood.pdf",
        "file_data": "base64encodedstring=="
    }]}
    res = client.post("/patients", json=payload)
    assert res.status_code == 201

    report = res.json()["patient"]["reports"][0]
    assert report["report_type"] == "Blood Test"
    assert report["report_id"].startswith("RPT-")
    assert "uploaded_at" in report


# ── READ ALL ─────────────────────────────────────────────────────────────────
def test_get_all_patients():
    res = client.get("/patients")
    assert res.status_code == 200
    assert "total" in res.json()
    assert isinstance(res.json()["patients"], list)


# ── READ ONE ─────────────────────────────────────────────────────────────────
def test_get_patient():
    patient = create_patient()
    uid = patient["uid"]

    res = client.get(f"/patients/{uid}")
    assert res.status_code == 200
    assert res.json()["uid"] == uid


def test_get_patient_not_found():
    res = client.get("/patients/PAT-INVALID")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# ── UPDATE ───────────────────────────────────────────────────────────────────
def test_update_patient_scalar_fields():
    patient = create_patient()
    uid = patient["uid"]

    res = client.put(f"/patients/{uid}", json={"age": 35, "contact": "8888888888"})
    assert res.status_code == 200

    updated = res.json()["patient"]
    assert updated["age"] == 35
    assert updated["contact"] == "8888888888"


def test_update_patient_appends_reports():
    patient = create_patient()
    uid = patient["uid"]

    # Add a report via update
    res = client.put(f"/patients/{uid}", json={"reports": [{
        "report_type": "X-Ray",
        "file_name": "xray.png",
        "file_data": "base64xray=="
    }]})
    assert res.status_code == 200

    reports = res.json()["patient"]["reports"]
    assert len(reports) == 1                         # appended, not replaced
    assert reports[0]["report_type"] == "X-Ray"


def test_update_patient_not_found():
    res = client.put("/patients/PAT-INVALID", json={"age": 40})
    assert res.status_code == 404


# ── DELETE ───────────────────────────────────────────────────────────────────
def test_delete_patient():
    patient = create_patient()
    uid = patient["uid"]

    res = client.delete(f"/patients/{uid}")
    assert res.status_code == 200
    assert res.json()["uid"] == uid

    # Confirm it's gone
    res = client.get(f"/patients/{uid}")
    assert res.status_code == 404


def test_delete_patient_not_found():
    res = client.delete("/patients/PAT-INVALID")
    assert res.status_code == 404


# ── CHAT ─────────────────────────────────────────────────────────────────────
def test_save_and_load_chat():
    patient = create_patient()
    uid = patient["uid"]

    # Post a user message
    res = client.post(f"/patients/{uid}/chat", json={"role": "user", "content": "Hello"})
    assert res.status_code == 200
    assert res.json()["status"] == "saved"

    # Post an assistant message
    client.post(f"/patients/{uid}/chat", json={"role": "assistant", "content": "Hi!"})

    # Load all messages
    res = client.get(f"/patients/{uid}/chat")
    assert res.status_code == 200

    messages = res.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_chat_patient_not_found():
    res = client.get("/patients/PAT-INVALID/chat")
    assert res.status_code == 404
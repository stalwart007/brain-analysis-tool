from fastapi.testclient import TestClient

from app import storage
from app.main import app
from app.validation import calibration_report

client = TestClient(app)


def _run(mean_intent: float, mean_engagement: float, actuals=None) -> dict:
    return {
        "result": {"mean_intent": mean_intent, "mean_engagement": mean_engagement},
        "actuals": actuals,
    }


def test_calibration_math():
    runs = [
        _run(0.8, 0.7, {"intent": 0.6, "engagement": 0.7}),  # intent err +0.2, eng 0
        _run(0.5, 0.4, {"intent": 0.6}),                     # intent err -0.1
        _run(0.9, 0.9),                                       # no actuals -> excluded
    ]
    report = calibration_report(runs)
    assert report.runs_with_actuals == 2
    assert report.intent is not None
    assert report.intent.n == 2
    assert abs(report.intent.mae - 0.15) < 1e-9
    assert abs(report.intent.bias - 0.05) < 1e-9  # slight over-prediction
    assert report.engagement is not None and report.engagement.n == 1


def test_calibration_empty():
    report = calibration_report([])
    assert report.runs_with_actuals == 0
    assert report.intent is None and report.engagement is None


def test_actuals_endpoint_roundtrip():
    run_id = storage.insert_swarm_run(
        {"scenario": "x"}, {"mean_intent": 0.7, "mean_engagement": 0.6}, kind="swarm"
    )

    res = client.post(f"/v1/runs/{run_id}/actuals", json={"intent": 0.5})
    assert res.status_code == 200

    report = client.get("/v1/validation/report").json()
    assert report["runs_with_actuals"] >= 1
    assert report["intent"]["n"] >= 1


def test_actuals_validation():
    run_id = storage.insert_swarm_run({}, {"mean_intent": 0.5}, kind="swarm")
    assert client.post(f"/v1/runs/{run_id}/actuals", json={}).status_code == 400
    assert (
        client.post(f"/v1/runs/{run_id}/actuals", json={"intent": 1.4}).status_code == 422
    )
    assert client.post("/v1/runs/nope/actuals", json={"intent": 0.5}).status_code == 404

import asyncio

import pytest

from app import jobs, storage
from app.main import app  # ensures DB init


@pytest.fixture
def stub_executor():
    """Replace all executors with a fast stub so no LLM is touched."""
    original = dict(jobs.EXECUTORS)

    async def ok(payload):
        return "swarm", {"mean_intent": 0.5, "mean_engagement": 0.5, "echo": payload}

    async def boom(payload):
        raise RuntimeError("simulated failure")

    jobs.EXECUTORS.clear()
    jobs.EXECUTORS.update({"swarm": ok, "compare": ok, "walk": ok, "batch_swarm": boom})
    yield
    jobs.EXECUTORS.clear()
    jobs.EXECUTORS.update(original)


def test_job_success_lifecycle(stub_executor):
    job_id = jobs.enqueue("swarm", {"scenario": "x"})
    assert storage.get_job(job_id)["status"] == "queued"

    asyncio.run(jobs._process(job_id))

    job = storage.get_job(job_id)
    assert job["status"] == "done"
    assert job["run_id"]
    run = storage.get_swarm_run(job["run_id"])
    assert run is not None
    assert run["request"]["job_id"] == job_id


def test_job_failure_is_captured_not_raised(stub_executor):
    job_id = jobs.enqueue("batch_swarm", {"scenario": "x"})
    asyncio.run(jobs._process(job_id))
    job = storage.get_job(job_id)
    assert job["status"] == "error"
    assert "simulated failure" in job["error"]
    # batch jobs record their mode on the stored request when they succeed;
    # on failure no run row exists
    assert job["run_id"] is None


def test_unknown_kind_rejected(stub_executor):
    with pytest.raises(ValueError):
        jobs.enqueue("nope", {})


def test_create_job_endpoint_validates_payload():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    # missing required 'scenario' for a swarm job -> 422, not 500
    res = client.post(
        "/v1/jobs", json={"kind": "batch_swarm", "payload": {"twins_per_persona": 5}}
    )
    assert res.status_code == 422
    # unknown kind -> 422 (schema enum on JobCreate.kind)
    assert (
        client.post("/v1/jobs", json={"kind": "bogus", "payload": {}}).status_code == 422
    )


def test_orphaned_running_jobs_are_requeued(stub_executor):
    job_id = jobs.enqueue("swarm", {"scenario": "y"})
    storage.update_job(job_id, "running")  # simulate a crash mid-run
    assert job_id in storage.queued_job_ids()


def test_done_jobs_are_not_reprocessed(stub_executor):
    job_id = jobs.enqueue("swarm", {"scenario": "z"})
    asyncio.run(jobs._process(job_id))
    run_id_first = storage.get_job(job_id)["run_id"]
    asyncio.run(jobs._process(job_id))  # second call must be a no-op
    assert storage.get_job(job_id)["run_id"] == run_id_first
    assert job_id not in storage.queued_job_ids()

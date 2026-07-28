from app.batch_swarm import build_batch_requests, parse_batch_line
from app.oai import strict_schema
from app.persona import seed_persona
from app.schemas import BehavioralSignal, TwinReaction


def _persona():
    return seed_persona(
        BehavioralSignal(
            deliberation="medium",
            frustration_signal="mild",
            exploration_style="scanner",
            price_sensitivity_signal="unknown",
            friction_hotspot=None,
            evidence="test",
            likely_mindset="just browsing",
            confidence=0.5,
        )
    )


def test_strict_schema_strips_unsupported_keywords():
    schema = strict_schema(TwinReaction)
    props = schema["properties"]
    # constraint keywords removed
    assert "minimum" not in props["engagement"]
    assert "maximum" not in props["engagement"]
    # strict-mode invariants enforced
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(props.keys())


def test_build_batch_requests_shape():
    requests = build_batch_requests([_persona(), _persona()], "An ad", 3, "high")
    assert len(requests) == 6
    ids = {r["custom_id"] for r in requests}
    assert len(ids) == 6  # unique custom_ids

    r0 = requests[0]
    assert r0["method"] == "POST"
    assert r0["url"] == "/v1/chat/completions"
    body = r0["body"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["temperature"] == 1.15  # high cognitive load
    # layering: preamble, scenario, persona+state, user instruction
    assert len(body["messages"]) == 4
    assert "An ad" in body["messages"][1]["content"]
    assert "exhausted" in body["messages"][2]["content"]  # high-load conditioning


def _line(status_code=200, content=None, refusal=None):
    return {
        "custom_id": "twin-0-0",
        "response": {
            "status_code": status_code,
            "body": {"choices": [{"message": {"content": content, "refusal": refusal}}]},
        },
    }


def test_parse_batch_line_success():
    payload = TwinReaction(
        engagement=0.4,
        intent_score=0.2,
        action="hesitate",
        dropoff_point=None,
        friction_notes="wall of text",
        inner_monologue="ugh, too long",
    ).model_dump_json()
    reaction = parse_batch_line(_line(content=payload))
    assert reaction is not None
    assert reaction.action == "hesitate"
    assert reaction.intent_score == 0.2


def test_parse_batch_line_failures():
    assert parse_batch_line(_line(status_code=500)) is None
    assert parse_batch_line(_line(content="not json")) is None
    assert parse_batch_line(_line(content=None)) is None
    assert parse_batch_line(_line(content="{}", refusal="I can't help with that")) is None
    assert parse_batch_line({"custom_id": "x", "error": "boom"}) is None

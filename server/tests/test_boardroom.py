"""The white room: orchestration, isolation, and the placebo arm.

Runs with NO API key. The model layer is replaced at the `boardroom.async_client`
seam rather than at `_ask_member`, which is the whole point of these tests: the
central claim of this module is about what is and is not in an OUTBOUND PROMPT,
and a stub that replaces the function which builds the prompt cannot verify it.
So the fake client records every payload the engine would have sent, and the
isolation assertions run over all of them at once.

Every statement the fake returns is tagged `[rep=R round=K speaker=NAME]`, so a
statement's origin is readable wherever it later turns up. That is what makes the
placebo arm testable at the mechanism level instead of by its summary statistics:
a placebo member must be shown statements tagged with the DONOR's replicate, and
must never be shown one tagged with its own.
"""

import asyncio
import json
import re
import time

import openai
import pytest
from fastapi.testclient import TestClient

from app import boardroom
from app.schemas import RoomMemberSpec, RoomRequest

MOTION = "Freeze the platform migration for two quarters and ship the billing rewrite first."
#: Deliberately not a real research question — the assertion below is that it
#: reaches no prompt, per StudyRequest's "THE TWINS NEVER SEE IT".
RESEARCH_QUESTION = "QUESTIONMARKER does the CFO actually fold under pressure"
PRIVATE_MARKER = "PRIVATESECRET"
PRIVATE_POSITION = 0.1234

_NAMES = ("Alba", "Bo", "Cyd", "Dov", "Esi", "Fen", "Gil", "Hux", "Ivo")

#: Distinct in wording as well as in numbers, because `cast_diversity` reads
#: both — a fixture of "archetype 0 / archetype 1 / archetype 2" is degenerate by
#: the token-overlap measure and would make every healthy-cast assertion below
#: assert the opposite of what it says.
_ROLES = (
    "chief financial officer",
    "junior product manager",
    "head of platform engineering",
    "general counsel",
    "regional sales director",
    "support team lead",
    "board observer",
    "staff data scientist",
    "recruiting partner",
)
_ARCHETYPES = (
    "the one who was burned by a failed migration",
    "the optimist who over-promises delivery dates",
    "the engineer who knows where the bodies are buried",
    "the lawyer counting regulatory exposure",
    "the closer whose quota depends on shipping",
    "the person who answers the phone when it breaks",
    "the outsider with no operational skin in this",
    "the sceptic who wants the number recomputed",
    "the one who has to hire for whatever we choose",
)
_STAKES = (
    "a budget line she already defended to the board",
    "a promise he made to a customer in writing",
    "eighteen months of on-call rotations",
    "a consent decree with a hard deadline",
    "a compensation plan tied to Q3 bookings",
    "a queue that doubles the week after any release",
    "a reputation for spotting the mistake early",
    "a forecast published under her own name",
    "three open requisitions and no story to tell",
)


def _member(i: int) -> RoomMemberSpec:
    return RoomMemberSpec(
        name=_NAMES[i],
        role=_ROLES[i],
        archetype=_ARCHETYPES[i],
        stake=_STAKES[i],
        system_prompt=f"You are person number {i}. You remember the outage in 201{i}.",
        openness=0.1 + 0.1 * i,
        assertiveness=0.9 - 0.1 * i,
        seniority=0.2 + 0.08 * i,
        risk_appetite=0.05 + 0.1 * i,
        expertise=[f"domain{i}"],
    )


def _cast(seats: int) -> list[RoomMemberSpec]:
    return [_member(i) for i in range(seats)]


def _request(seats=3, rounds=2, replicates=2, placebo=1, **kw) -> RoomRequest:
    return RoomRequest(
        motion=MOTION,
        research_question=RESEARCH_QUESTION,
        members=_cast(seats),
        seats=seats,
        rounds=rounds,
        replicates=replicates,
        placebo_replicates=placebo,
        **kw,
    )


# ────────────────────────────── the fake model layer ──────────────────────────


class _Message:
    def __init__(self, content):
        self.content = content
        self.refusal = None


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeRoom:
    """A deterministic, offline model layer that records what it was asked.

    The replicate index is recovered from the **seed**, by inverting
    `boardroom._call_seed`, which is unique per (motion, replicate, round,
    member) by construction.

    It used to be recovered by COUNTING — `call // (seats * (1 + rounds))` —
    which silently assumed "no replicate begins before the previous one has
    finished". That assumption held only while the engine ran replicates one
    after another, and it broke the moment they were made concurrent: canned
    responses landed on the wrong replicate and two tests failed with
    cross-contamination that did not exist in the engine. Inverting the seed is
    schedule-agnostic, so the double no longer encodes a scheduling decision it
    has no business knowing about, and it is STRICTLY STRONGER: it now also
    catches a wrong or colliding seed, which the counter could not see at all.

    `test_event_sequence_is_well_formed` still cross-checks the tag the fake
    stamped against the replicate the engine reported for the same turn.
    """

    def __init__(
        self,
        seats: int,
        rounds: int,
        fail: frozenset[str] = frozenset(),
        motion: str = MOTION,
    ):
        self.seats = seats
        self.rounds = rounds
        self.fail = fail
        self.prompts: list[dict] = []
        self.cast_calls = 0
        self.turn_calls = 0
        # seed → (replicate, round, member), built well PAST the schema's own
        # ceilings (9 seats, 8 rounds, 8+4 replicates) rather than at exactly the
        # dimensions this instance was constructed with. Several tests
        # deliberately hand the fake different numbers from the request — a
        # mismatch the old counting inference silently absorbed — and the map is
        # only an inverse lookup, so over-building costs nothing and a genuine
        # unrecognised seed still fails loudly. `_call_seed`'s offsets stay far
        # below 2**31 over this range, so there is no wraparound and the
        # collision assertion below is a real check rather than decoration.
        self._by_seed: dict[int, tuple[int, int, int]] = {}
        for rep in range(20):
            for rnd in range(14):
                for member in range(len(_NAMES)):
                    key = boardroom._call_seed(motion, rep, rnd, member)
                    assert key not in self._by_seed, (
                        f"_call_seed collision: {(rep, rnd, member)} vs "
                        f"{self._by_seed[key]} — the seed is supposed to be "
                        "unique per slot"
                    )
                    self._by_seed[key] = (rep, rnd, member)

    # -- the client surface the engine actually uses ------------------------
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(
        self, *, model, messages, temperature=None, seed=None, response_format=None, **kw
    ):
        name = response_format["json_schema"]["name"]
        if name == "RoomCast":
            self.cast_calls += 1
            return _Completion(json.dumps(self._cast_payload()))

        self.turn_calls += 1
        speaker = _speaker_of(messages)
        round_index = _round_of(messages)
        assert seed in self._by_seed, (
            f"unrecognised call seed {seed} — the engine seeded a slot outside "
            "the schedule, or _call_seed changed"
        )
        replicate, seeded_round, seeded_member = self._by_seed[seed]
        # The seed says which slot this is; the prompt says which slot the member
        # thinks it is. They must agree, or the engine has crossed its wires.
        assert seeded_round == round_index, (seed, seeded_round, round_index)
        assert _NAMES.index(speaker) == seeded_member, (seed, speaker, seeded_member)
        self.prompts.append(
            {
                "replicate": replicate,
                "round": round_index,
                "speaker": speaker,
                "messages": messages,
                "seed": seed,
                "model": model,
            }
        )
        if speaker in self.fail:
            # The exception that bites in practice: openai.OpenAIError is the
            # ROOT of the hierarchy and what AsyncOpenAI() raises with no key,
            # so a handler that only catches APIError misses it entirely.
            raise openai.OpenAIError("no API key configured")
        return _Completion(json.dumps(self._turn_payload(replicate, round_index, speaker)))

    # -- payloads -----------------------------------------------------------
    def _cast_payload(self) -> dict:
        return {"members": [m.model_dump() for m in _cast(self.seats)]}

    def _turn_payload(self, replicate: int, round_index: int, speaker: str) -> dict:
        index = _NAMES.index(speaker)
        return {
            "public_statement": (
                f"[rep={replicate} round={round_index} speaker={speaker}] "
                "The billing rewrite slipped twice already and I am not signing again."
            ),
            "public_position": round(0.15 + 0.12 * index + 0.05 * round_index, 3),
            "private_position": PRIVATE_POSITION,
            "private_note": (
                f"[{PRIVATE_MARKER} rep={replicate} round={round_index} "
                f"speaker={speaker}] I will fold if Alba pushes."
            ),
            "conceded_to": ["Alba"] if speaker != "Alba" else [],
        }


def _speaker_of(messages: list[dict]) -> str:
    """The conditioning message opens 'You are NAME — role'."""
    match = re.search(r"You are ([^—]+) —", messages[1]["content"])
    assert match, messages[1]["content"][:120]
    return match.group(1).strip()


def _round_of(messages: list[dict]) -> int:
    user = messages[-1]["content"]
    if "SEALED BALLOT" in user:
        return 0
    match = re.search(r"in round (\d+) of", user)
    assert match, user[:200]
    return int(match.group(1))


def _table_of(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    match = re.search(r"<table>(.*?)</table>", user, re.S)
    return match.group(1) if match else ""


def _own_of(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    match = re.search(r"What you have said so far:(.*?)\n\nYou are speaking", user, re.S)
    return match.group(1) if match else ""


def _install(monkeypatch, fake: FakeRoom, analysis=None) -> FakeRoom:
    monkeypatch.setattr(boardroom, "async_client", lambda: fake)
    # The analysis seam is stubbed by default: these tests are about
    # orchestration and isolation, and a failure in `room`'s estimators should
    # not present as a broken event stream. The real module is exercised without
    # a stub by `test_the_real_analysis_runs_end_to_end`, and its signature is
    # pinned by `test_analysis_payload_matches_room_analysis`.
    monkeypatch.setattr(
        boardroom,
        "_analyse",
        analysis or (lambda payload: {"stub": True, "turns_seen": len(payload["turns"])}),
    )
    return fake


def collect(gen) -> list[dict]:
    async def run():
        return [e async for e in gen]

    return asyncio.run(asyncio.wait_for(run(), timeout=20.0))


# ───────────────────────────── the isolation wall ─────────────────────────────


def test_private_text_never_enters_any_prompt(monkeypatch):
    """The claim the whole module rests on, asserted over every outbound payload.

    Three things must never appear in a prompt: another member's private note,
    ANY private position (not even the member's own — see the module docstring's
    point 1), and the researcher's question, which `StudyRequest` documents must
    never reach the simulated speakers because a room told what the researcher
    hopes to find will find it.
    """
    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    events = collect(boardroom.stream_deliberation(_request(seats=3, rounds=2)))

    assert events[-1]["type"] == "done"
    assert fake.prompts, "no prompts were captured — the seam moved"

    for record in fake.prompts:
        blob = json.dumps(record["messages"])
        assert PRIVATE_MARKER not in blob, (
            f"a private note reached {record['speaker']} at round {record['round']}"
        )
        assert str(PRIVATE_POSITION) not in blob, (
            "a private position was rendered into a prompt"
        )
        assert "QUESTIONMARKER" not in blob, "the research question reached the table"

    # And the private half really does exist — otherwise the assertions above
    # would pass on an engine that simply never produced any private text.
    private = {t["private_note"] for t in events[-1]["result"]["turns"]}
    assert private and all(PRIVATE_MARKER in note for note in private)
    assert {t["private_position"] for t in events[-1]["result"]["turns"]} == {
        PRIVATE_POSITION
    }


def test_every_prompt_restates_the_motion_polarity(monkeypatch):
    """A regression guard for a defect found in the first live run, not a
    hypothetical: a CFO argued FOR freezing a migration and scored himself 0.20 —
    against — because the motion's own verb is a negative action and "for the
    motion" then reads as "for the thing being stopped". Positions that do not
    track the words they came with are worse than no positions, since every
    estimator downstream reads the number. The orientation is therefore restated
    at the point of scoring in EVERY turn, not once in the frozen preamble."""
    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    collect(boardroom.stream_deliberation(_request(seats=3, rounds=2)))

    assert fake.prompts
    for record in fake.prompts:
        user = record["messages"][-1]["content"]
        assert "WHETHER THIS MOTION SHOULD PASS" in user
        assert "you are HIGH on this scale, not low" in user
        assert MOTION in user


def test_no_position_number_is_ever_shown_to_the_table(monkeypatch):
    """Members hear arguments, not scalars. Rendering the position scale would
    let the room herd on the exact number being estimated."""
    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    events = collect(boardroom.stream_deliberation(_request(seats=3, rounds=2)))

    shown = {
        round(t["public_position"], 3) for t in events[-1]["result"]["turns"]
    }
    assert len(shown) > 1, "the fake must vary public_position for this to mean anything"
    for record in fake.prompts:
        table = _table_of(record["messages"])
        for value in shown:
            assert str(value) not in table


def test_round_zero_is_a_sealed_ballot(monkeypatch):
    """Round 0 is recorded and never transmitted. If its statements were
    circulated at round 1, the first speaker of round 1 would no longer be
    speaking blind and the speaking-order effect — the thing replicates exist to
    average over — would not be measurable at all."""
    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    collect(boardroom.stream_deliberation(_request(seats=3, rounds=2)))

    ballots = [r for r in fake.prompts if r["round"] == 0]
    assert len(ballots) == 3 * 3  # 3 seats × 3 replicates
    for record in ballots:
        assert "SEALED BALLOT" in record["messages"][-1]["content"]
        assert "<table>" not in record["messages"][-1]["content"]

    for record in fake.prompts:
        assert "round=0" not in _table_of(record["messages"]), (
            "a round-0 ballot statement was transmitted to the table"
        )


def test_a_member_hears_earlier_speakers_and_not_later_ones(monkeypatch):
    """Sequential-within-round is what makes a cascade real: speaker j sees
    exactly the j−1 members ahead of them this round, plus every earlier round."""
    fake = _install(monkeypatch, FakeRoom(seats=4, rounds=2))
    events = collect(boardroom.stream_deliberation(_request(seats=4, rounds=2, replicates=1, placebo=0)))

    order = next(e for e in events if e["type"] == "replicate_start")[
        "speaking_order_names"
    ]

    for record in fake.prompts:
        if record["round"] != 1:
            continue
        position = order.index(record["speaker"])
        table = _table_of(record["messages"])
        heard = set(re.findall(r"speaker=(\w+)", table))
        assert heard == set(order[:position]), (
            f"{record['speaker']} spoke {position} and heard {heard}"
        )
        assert record["speaker"] not in heard


# ────────────────────────────── the placebo arm ───────────────────────────────


def test_placebo_arm_shows_foreign_statements(monkeypatch):
    """The critical control, asserted at the mechanism.

    A placebo member must be shown statements from the DONOR replicate in
    exactly the slots the real arm would have filled from its own, while their
    own history stays their own — nobody is ever told they said something they
    did not say. That is what makes it a yoked control rather than a different
    experiment: the only thing removed is the causal link between what this
    member said and what comes back.
    """
    seats, rounds, replicates = 3, 2, 2
    fake = _install(monkeypatch, FakeRoom(seats=seats, rounds=rounds))
    events = collect(
        boardroom.stream_deliberation(
            _request(seats=seats, rounds=rounds, replicates=replicates, placebo=1)
        )
    )

    starts = [e for e in events if e["type"] == "replicate_start"]
    assert [e["placebo"] for e in starts] == [False, False, True]
    assert [e["donor_replicate"] for e in starts] == [None, None, 0]

    placebo_prompts = [
        r for r in fake.prompts if r["replicate"] == 2 and r["round"] >= 1
    ]
    assert placebo_prompts

    saw_foreign = False
    for record in placebo_prompts:
        table = _table_of(record["messages"])
        origins = set(re.findall(r"rep=(\d+) round", table))
        if origins:
            saw_foreign = True
            assert origins == {"0"}, (
                f"{record['speaker']} heard statements from {origins}, not the donor"
            )
        own = _own_of(record["messages"])
        own_origins = set(re.findall(r"rep=(\d+) round", own))
        assert own_origins <= {"2"}, (
            f"{record['speaker']} was shown a foreign version of their own words"
        )
    assert saw_foreign, "the placebo arm never showed anybody anything"

    # The real arms are not yoked: they hear their own replicate and nothing else.
    for record in fake.prompts:
        if record["replicate"] >= replicates or record["round"] < 1:
            continue
        origins = set(re.findall(r"rep=(\d+) round", _table_of(record["messages"])))
        assert origins <= {str(record["replicate"])}


def test_placebo_hears_the_same_slots_as_the_real_arm(monkeypatch):
    """Same shape, foreign words. If the placebo showed a different NUMBER of
    statements — or statements from rounds that had not happened yet — the two
    arms would differ in more than the one thing being controlled, and any
    difference between them would be uninterpretable."""
    seats, rounds = 3, 2
    fake = _install(monkeypatch, FakeRoom(seats=seats, rounds=rounds))
    collect(
        boardroom.stream_deliberation(
            _request(seats=seats, rounds=rounds, replicates=1, placebo=1)
        )
    )
    heard_counts: dict[int, list[int]] = {0: [], 1: []}
    for record in fake.prompts:
        if record["round"] < 1:
            continue
        table = _table_of(record["messages"])
        heard_counts[record["replicate"]].append(len(re.findall(r"speaker=", table)))
    assert heard_counts[0] == heard_counts[1] != []
    # …and the donor's future never leaks: nothing from a round not yet reached.
    for record in fake.prompts:
        if record["replicate"] != 1 or record["round"] < 1:
            continue
        rounds_shown = {int(r) for r in re.findall(r"round=(\d+)", _table_of(record["messages"]))}
        assert all(r <= record["round"] for r in rounds_shown)


# ──────────────────────────── seating and determinism ─────────────────────────


def test_one_semaphore_bounds_the_whole_run(monkeypatch):
    """ONE semaphore for the generator, threaded through — never one per
    replicate or per round.

    The audit backlog's infrastructure section is explicit that a per-subgroup
    semaphore multiplies the concurrency ceiling by the fan-out. Round 0 is the
    only concurrent phase here, and it runs once per replicate, so a semaphore
    created inside that loop would look correct at every call site and quietly
    allow `replicates × SWARM_CONCURRENCY` in flight. This measures the ceiling
    instead of reading the code.
    """
    monkeypatch.setattr(boardroom, "SWARM_CONCURRENCY", 2)

    class Counting(FakeRoom):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.in_flight = 0
            self.peak = 0

        async def create(self, **kw):
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                # Yield control so genuinely-concurrent calls overlap here.
                await asyncio.sleep(0)
                return await super().create(**kw)
            finally:
                self.in_flight -= 1

    fake = _install(monkeypatch, Counting(seats=6, rounds=2))
    events = collect(
        boardroom.stream_deliberation(_request(seats=6, rounds=2, replicates=2, placebo=1))
    )
    assert events[-1]["type"] == "done"
    assert fake.peak <= 2, f"{fake.peak} calls were in flight against a ceiling of 2"
    assert fake.peak == 2, "round 0 must actually run concurrently"


def test_speaking_order_is_deterministic_per_seed_and_differs_per_replicate():
    first = boardroom.speaking_orders(5, 4, seed=99)
    again = boardroom.speaking_orders(5, 4, seed=99)
    other = boardroom.speaking_orders(5, 4, seed=100)

    assert first == again, "same seed must give the same schedule, forever"
    assert first != other, "the seed must actually do something"
    assert all(sorted(o) == list(range(5)) for o in first)
    # Distinct orders are enforced, not left to chance: a replicate that walks an
    # order already walked is a wasted replicate, and independent draws collide
    # around 5% of the time at these sizes.
    assert len({tuple(o) for o in first}) == 4

    # More replicates than permutations cannot be distinct; it must not hang.
    tiny = boardroom.speaking_orders(2, 5, seed=3)
    assert len(tiny) == 5 and all(sorted(o) == [0, 1] for o in tiny)


def test_order_seed_is_stable_per_motion_and_differs_across_motions():
    assert boardroom.order_seed(MOTION) == boardroom.order_seed(f"  {MOTION}  ")
    assert boardroom.order_seed(MOTION) != boardroom.order_seed(MOTION + " Or not.")


def test_replicates_walk_different_orders_in_a_real_run(monkeypatch):
    fake = _install(monkeypatch, FakeRoom(seats=5, rounds=2))
    events = collect(
        boardroom.stream_deliberation(_request(seats=5, rounds=2, replicates=3, placebo=0))
    )
    orders = [tuple(e["speaking_order"]) for e in events if e["type"] == "replicate_start"]
    assert len(orders) == 3 and len(set(orders)) == 3


# ───────────────────────────── the event contract ─────────────────────────────


def test_event_sequence_is_well_formed(monkeypatch):
    seats, rounds, replicates, placebo = 3, 2, 2, 1
    fake = _install(monkeypatch, FakeRoom(seats=seats, rounds=rounds))
    events = collect(
        boardroom.stream_deliberation(
            _request(seats=seats, rounds=rounds, replicates=replicates, placebo=placebo)
        )
    )
    kinds = [e["type"] for e in events]

    assert kinds[0] == "start"
    assert kinds[1] == "cast"
    assert kinds[-1] == "done"
    assert kinds[-2] == "stage" and events[-2]["stage"] == "analysis"

    total = replicates + placebo
    assert events[0]["total"] == seats * (1 + rounds) * total
    assert events[0]["members"] == list(_NAMES[:seats])
    assert kinds.count("replicate_start") == total
    assert kinds.count("replicate_done") == total
    assert kinds.count("round_done") == total * (1 + rounds)
    assert kinds.count("turn") == seats * (1 + rounds) * total
    assert kinds.count("turn_failed") == 0

    # Per replicate: replicate_start, then every round's turns followed by that
    # round's round_done, then replicate_done.
    for replicate in range(total):
        own = [
            e
            for e in events
            if e.get("replicate") == replicate and e["type"] != "replicate_skipped"
        ]
        assert own[0]["type"] == "replicate_start"
        assert own[-1]["type"] == "replicate_done"
        expected_round = 0
        seen_in_round = 0
        for evt in own[1:-1]:
            if evt["type"] == "turn":
                assert evt["round"] == expected_round
                assert evt["speaking_index"] == seen_in_round or expected_round == 0
                seen_in_round += 1
            else:
                assert evt["type"] == "round_done" and evt["round"] == expected_round
                assert evt["spoke"] == seats
                expected_round += 1
                seen_in_round = 0
        assert expected_round == rounds + 1

    # No live frame carries the private half — one assertion for the whole
    # stream, which is why it is withheld wholesale rather than field by field.
    for evt in events[:-1]:
        blob = json.dumps(evt)
        assert PRIVATE_MARKER not in blob
        assert "private_position" not in blob
        assert "private_note" not in blob

    # The counter the fake uses to tag statements is not taken on trust: every
    # emitted turn's statement must carry the replicate the engine reported.
    for evt in [e for e in events if e["type"] == "turn"]:
        tag = re.search(r"rep=(\d+) round=(\d+) speaker=(\w+)", evt["statement"])
        assert tag, evt["statement"]
        assert int(tag.group(1)) == evt["replicate"]
        assert int(tag.group(2)) == evt["round"]
        assert tag.group(3) == evt["member"]


def test_terminal_result_carries_the_measurement(monkeypatch):
    seen: dict = {}

    def analysis(payload):
        seen.update(payload)
        return {"stub": True}

    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=2), analysis=analysis)
    events = collect(
        boardroom.stream_deliberation(_request(seats=3, rounds=2, replicates=2, placebo=1))
    )
    result = events[-1]["result"]

    assert result["run_id"] == ""  # stamped by storage on insert, like every study
    assert result["seats"] == 3 and result["rounds"] == 2
    assert result["turns_failed"] == 0
    assert len(result["turns"]) == 3 * 3 * 3
    assert len(result["speaking_orders"]) == 3
    assert result["analysis"] == {"stub": True}
    # The cast arrived already assembled with no brief, so its origin is not
    # visible from here — "approved", not "described" (which would credit prose
    # that was never sent) and not "inferred" (which would assert something
    # unknown). Never "observed" in any branch.
    assert result["cast_provenance"]["provenance"] == "approved"
    assert "private_position" in result["turns"][0]
    assert {t["placebo"] for t in result["turns"]} == {False, True}

    # …and the analysis sees the whole transcript, private half included.
    assert len(seen["turns"]) == 27
    assert all("private_position" in t for t in seen["turns"])
    assert [m["arm"] for m in seen["replicates_meta"]] == ["real", "real", "placebo"]
    assert [m["order"] for m in seen["replicates_meta"]] == [
        [_NAMES[i] for i in order] for order in result["speaking_orders"]
    ]


def test_conceded_to_is_resolved_against_the_actual_place_cards(monkeypatch):
    """The fake has everyone concede to 'Alba'. A name nobody at the table has
    would be neither silently kept nor silently dropped."""
    fake = _install(monkeypatch, FakeRoom(seats=3, rounds=1))
    events = collect(
        boardroom.stream_deliberation(_request(seats=3, rounds=2, replicates=1, placebo=0))
    )
    turns = events[-1]["result"]["turns"]
    assert any(t["conceded_to"] == ["Alba"] for t in turns)
    # Alba never concedes to herself even though the stub would allow it.
    assert all(t["conceded_to"] == [] for t in turns if t["member"] == "Alba")
    assert all(t["conceded_to_unattributable"] == [] for t in turns)


# ─────────────────────── the instrument-health precondition ───────────────────


def _turn(round_index, member, public, private, placebo=False):
    return {
        "round": round_index,
        "member": member,
        "public_position": public,
        "private_position": private,
        "placebo": placebo,
    }


def test_a_pinned_scale_is_reported_as_unusable():
    """The failure the first live run of this module actually had: every position
    at 0.00 from round 1 on, in both arms. Nothing was broken mechanically and
    every downstream statistic was computable — convergence perfect, gap exactly
    zero, placebo matching the real arm — because the scale had no room to move."""
    pinned = (
        [_turn(0, n, 0.0, 0.0) for n in "ABC"]
        + [_turn(1, n, 0.0, 0.0) for n in "ABC"]
        + [_turn(0, n, 0.0, 0.0, placebo=True) for n in "ABC"]
    )
    check = boardroom.instrument_check(pinned)
    assert check["saturated"] is True
    assert check["usable"] is False
    assert check["overall"]["at_floor"] == 1.0
    assert check["overall"]["distinct_public_positions"] == 1
    assert check["real_arm"]["turns"] == 6 and check["placebo_arm"]["turns"] == 3
    assert "no room left to move" in check["note"]


def test_a_room_that_agreed_before_it_spoke_is_reported_separately():
    """A different failure with a different fix: the scale is fine, the CAST had
    no opposition. Deliberation cannot move a room that already agrees, so a flat
    trajectory says nothing about influence."""
    agreed = [
        _turn(0, "A", 0.7, 0.72),
        _turn(0, "B", 0.6, 0.66),
        _turn(0, "C", 0.8, 0.81),
        _turn(1, "A", 0.72, 0.7),
        _turn(1, "B", 0.65, 0.6),
        _turn(1, "C", 0.9, 0.55),
    ]
    check = boardroom.instrument_check(agreed)
    assert check["no_opposition_at_ballot"] is True
    assert check["saturated"] is False
    assert check["usable"] is False
    assert "before anyone spoke" in check["note"]

    split = agreed + [_turn(0, "D", 0.2, 0.18), _turn(1, "D", 0.25, 0.2)]
    healthy = boardroom.instrument_check(split)
    assert healthy["no_opposition_at_ballot"] is False
    assert healthy["usable"] is True
    assert healthy["note"] is None
    assert healthy["overall"]["mean_abs_gap"] > 0


def test_instrument_check_rides_with_every_result(monkeypatch):
    _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    events = collect(boardroom.stream_deliberation(_request(seats=3, rounds=2)))
    check = events[-1]["result"]["instrument_check"]
    assert check["overall"]["turns"] == 27
    assert check["real_arm"]["turns"] == 18 and check["placebo_arm"]["turns"] == 9
    assert isinstance(check["usable"], bool)


def test_instrument_check_survives_an_empty_transcript():
    empty = boardroom.instrument_check([])
    assert empty["usable"] is True and empty["note"] is None
    assert empty["overall"] == {"turns": 0}


# ────────────────────────────── failure handling ──────────────────────────────


def test_one_failing_member_does_not_hang_or_kill_the_run(monkeypatch):
    """A member who fails is silent for that turn and the room carries on.

    The `stream_walkthrough` bug in the audit backlog was a generator that
    counted its own completions and therefore hung forever on an exception. This
    generator awaits its sequential turns inline and catches per-call failures in
    `_ask_member`, so there is no completion counter to desynchronise — this test
    is what keeps that true, under a wait_for so a regression fails fast rather
    than wedging the suite.
    """
    seats, rounds = 3, 2
    fake = _install(
        monkeypatch, FakeRoom(seats=seats, rounds=rounds, fail=frozenset({"Bo"}))
    )
    events = collect(
        boardroom.stream_deliberation(
            _request(seats=seats, rounds=rounds, replicates=1, placebo=1)
        )
    )

    assert events[-1]["type"] == "done"
    failures = [e for e in events if e["type"] == "turn_failed"]
    assert len(failures) == (1 + rounds) * 2  # Bo, every round, both arms
    assert {e["member"] for e in failures} == {"Bo"}
    assert events[-1]["result"]["turns_failed"] == len(failures)
    assert all(t["member"] != "Bo" for t in events[-1]["result"]["turns"])

    # The room still deliberated: the survivors heard each other, and a round
    # where somebody was silent reports the count that spoke rather than the
    # count that was seated.
    partial = [e for e in events if e["type"] == "round_done"]
    assert all(e["spoke"] == seats - 1 for e in partial)
    assert all(e["public_mean"] is not None for e in partial)

    # Per replicate: dispatched minus survivors, not a count of what came back.
    # A room where a third of the members failed must not read as a healthy
    # smaller room — the same defect the swarm aggregator guards against.
    for evt in [e for e in events if e["type"] == "replicate_done"]:
        assert evt["turns"] == (seats - 1) * (1 + rounds)
        assert evt["failed"] == 1 + rounds
        assert evt["rounds_completed"] == rounds
    # Bo was still attempted every time — a failure is not a reason to stop
    # asking, since the next call may succeed.
    assert sum(1 for p in fake.prompts if p["speaker"] == "Bo") == (1 + rounds) * 2


def test_a_room_where_everybody_fails_terminates_with_an_error(monkeypatch):
    """No hang, and no empty-looking success.

    `_sse` turns the RuntimeError into an `error` frame, which is the correct
    outcome: what the backlog entry was about is that control never reached here
    at all. The placebo arm is reported as skipped rather than silently dropped,
    because a missing control changes how the rest may be read — and here there
    is no rest.
    """
    fake = _install(
        monkeypatch,
        FakeRoom(seats=3, rounds=2, fail=frozenset(_NAMES)),
    )
    events: list[dict] = []

    async def drain():
        async for evt in boardroom.stream_deliberation(
            _request(seats=3, rounds=2, replicates=1, placebo=1)
        ):
            events.append(evt)

    with pytest.raises(RuntimeError, match="no statements at all"):
        asyncio.run(asyncio.wait_for(drain(), timeout=20.0))

    assert [e["type"] for e in events].count("turn_failed") == 9
    skipped = [e for e in events if e["type"] == "replicate_skipped"]
    assert len(skipped) == 1 and skipped[0]["placebo"] is True
    assert skipped[0]["donor_replicate"] == 0


def test_a_failed_cast_call_does_not_start_a_room(monkeypatch):
    class Exploding(FakeRoom):
        async def create(self, **kw):
            raise openai.OpenAIError("no API key configured")

    _install(monkeypatch, Exploding(seats=3, rounds=2))
    request = _request(seats=3, rounds=2)
    request.members = []

    async def drain():
        return [e async for e in boardroom.stream_deliberation(request)]

    with pytest.raises(openai.OpenAIError):
        asyncio.run(asyncio.wait_for(drain(), timeout=20.0))


# ───────────────────────────── casting the room ───────────────────────────────


def test_cast_room_infers_a_room_and_labels_it(monkeypatch):
    fake = _install(monkeypatch, FakeRoom(seats=4, rounds=1))
    members = asyncio.run(boardroom.cast_room(MOTION, "", 4))
    assert len(members) == 4
    assert fake.cast_calls == 1

    inferred = boardroom.cast_provenance("", members)
    described = boardroom.cast_provenance("a skeptical CFO and a junior PM", members)
    assert inferred["provenance"] == "inferred"
    assert described["provenance"] == "described"
    # Never "observed", in either branch: nobody in a room is derived from
    # telemetry, and a researcher describing a character in more detail does not
    # turn an assumption into a measurement.
    assert "measured" in inferred["note"] or "measured" in inferred["note"].lower()
    assert "observation" in described["note"]


def test_cast_room_disambiguates_duplicate_names(monkeypatch):
    class Twins(FakeRoom):
        def _cast_payload(self):
            payload = super()._cast_payload()
            for member in payload["members"]:
                member["name"] = "Dana"
            return payload

    _install(monkeypatch, Twins(seats=3, rounds=1))
    members = asyncio.run(boardroom.cast_room(MOTION, "", 3))
    names = [m.name for m in members]
    assert len(set(names)) == 3 and names[0] == "Dana"


def test_cast_room_clamps_out_of_range_dispositions(monkeypatch):
    """`oai._clean` strips minimum/maximum from the wire schema, so a model may
    legally return 1.4 for a 0-1 disposition."""

    class OutOfRange(FakeRoom):
        def _cast_payload(self):
            payload = super()._cast_payload()
            payload["members"][0]["openness"] = 1.4
            payload["members"][1]["risk_appetite"] = -0.6
            return payload

    _install(monkeypatch, OutOfRange(seats=3, rounds=1))
    members = asyncio.run(boardroom.cast_room(MOTION, "", 3))
    assert members[0].openness == 1.0
    assert members[1].risk_appetite == 0.0


def test_a_degenerate_cast_is_flagged_not_refused():
    """Everyone the same archetype, flat dispositions: the room cannot disagree.
    Surfaced with the numbers behind it, and the run is still allowed — 'this
    room cannot disagree with itself' is a finding, as long as nobody reads its
    consensus as one."""
    clones = [
        RoomMemberSpec(
            name=_NAMES[i],
            role="finance lead",
            archetype="cautious finance person",
            stake="the budget they own",
            system_prompt="You are careful with money.",
            openness=0.5,
            assertiveness=0.5,
            seniority=0.5,
            risk_appetite=0.5,
            expertise=["finance"],
        )
        for i in range(4)
    ]
    verdict = boardroom.cast_diversity(clones)
    assert verdict["degenerate"] is True
    assert verdict["distinct_archetypes"] == 1
    assert verdict["disposition_range"]["openness"] == 0.0
    assert verdict["mean_archetype_overlap"] == 1.0
    assert "agree by construction" in verdict["note"]

    healthy = boardroom.cast_diversity(_cast(5))
    assert healthy["degenerate"] is False
    assert healthy["note"] is None
    assert healthy["distinct_archetypes"] == 5


def test_diversity_catches_a_renamed_cast_with_spread_numbers():
    """The subtler degeneracy: different job titles, same person underneath.
    Distinct archetype STRINGS pass an exact-match count, so the token overlap is
    what has to catch it."""
    rebadged = [
        RoomMemberSpec(
            name=_NAMES[i],
            role=f"cautious finance lead {i}",
            archetype=f"cautious finance person {i}",
            stake=f"the finance budget they own {i}",
            system_prompt="You are careful with money.",
            openness=0.1 + 0.2 * i,
            assertiveness=0.9 - 0.2 * i,
            seniority=0.2 * i,
            risk_appetite=0.15 * i,
            expertise=["finance"],
        )
        for i in range(4)
    ]
    verdict = boardroom.cast_diversity(rebadged)
    assert verdict["distinct_archetypes"] == 4
    assert verdict["degenerate"] is True
    assert "same words" in verdict["note"]


# ────────────────────────────── budget and endpoints ──────────────────────────


def test_room_cost_is_seats_times_rounds_times_replicates():
    request = _request(seats=3, rounds=4, replicates=3, placebo=1)
    # 3 seats × (1 sealed ballot + 4 rounds) × (3 real + 1 placebo)
    assert boardroom.room_call_estimate(request) == 3 * 5 * 4 == 60

    # An approved cast overrides `seats`, because that is the cast that runs.
    request.members = _cast(6)
    assert boardroom.room_call_estimate(request) == 6 * 5 * 4

    # The schema caps: 9 × (1 + 8) × (8 + 4).
    worst = RoomRequest(
        motion=MOTION, seats=9, rounds=8, replicates=8, placebo_replicates=4
    )
    assert boardroom.room_call_estimate(worst) == 972


def test_estimate_twin_calls_routes_a_room_through_its_own_cost_model():
    """The generic estimator multiplies by the persona window, which a room does
    not have — so without the branch a 972-call room scores as 0 and walks past
    the ceiling."""
    from app.main import estimate_twin_calls

    request = RoomRequest(
        motion=MOTION, seats=5, rounds=4, replicates=3, placebo_replicates=1
    )
    assert estimate_twin_calls(0, request) == 100
    # The persona count is not a multiplier here, at any value.
    assert estimate_twin_calls(200, request) == 100


def _client():
    from app.main import app

    return TestClient(app)


def test_an_oversized_room_is_refused_with_a_413(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_TWINS_PER_RUN", 50)
    response = _client().post(
        "/v1/room/stream",
        json={
            "motion": MOTION,
            "seats": 9,
            "rounds": 8,
            "replicates": 8,
            "placebo_replicates": 4,
        },
    )
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "972 twin calls" in detail and "ceiling" in detail
    # The advice names the controls that actually move the number — not the
    # persona window, which a room does not have.
    assert "replicates" in detail and "rounds" in detail
    assert "personas are currently in scope" not in detail


def test_a_room_needs_three_seats_and_unique_names():
    two = [m.model_dump() for m in _cast(2)]
    response = _client().post(
        "/v1/room/stream", json={"motion": MOTION, "members": two}
    )
    assert response.status_code == 400
    assert "three characters" in response.json()["detail"]

    clashing = [m.model_dump() for m in _cast(3)]
    clashing[1]["name"] = clashing[0]["name"]
    response = _client().post(
        "/v1/room/stream", json={"motion": MOTION, "members": clashing}
    )
    assert response.status_code == 400
    assert "unique" in response.json()["detail"]


def test_cast_endpoint_returns_a_cast_for_approval(monkeypatch):
    _install(monkeypatch, FakeRoom(seats=4, rounds=1))
    response = _client().post(
        "/v1/room/cast",
        json={"motion": MOTION, "cast_brief": "a burned CFO, an eager PM", "seats": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["members"]) == 4
    assert body["provenance"]["provenance"] == "described"
    assert body["diversity"]["degenerate"] is False
    # The approval step is also the budget step. Derived from the request model's
    # own defaults rather than hardcoded: this assertion previously baked in
    # `replicates=3` and broke the moment that default moved to 5, which told us
    # nothing about the formula it was meant to protect.
    defaults = RoomRequest(motion=MOTION)
    expected = 4 * (1 + defaults.rounds) * (defaults.replicates + defaults.placebo_replicates)
    assert body["estimated_calls"] == expected
    assert body["call_ceiling"] > 0


def test_cast_endpoint_reports_a_missing_key_as_503():
    """Key-free by construction: `AsyncOpenAI()` raises the root OpenAIError,
    which must surface as a 503 about configuration rather than a 500."""
    response = _client().post("/v1/room/cast", json={"motion": MOTION, "seats": 3})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_stream_endpoint_withholds_private_positions_until_done(monkeypatch):
    """End to end through `_sse`: the SSE body must contain the private half
    exactly once, in the terminal frame."""
    _install(monkeypatch, FakeRoom(seats=3, rounds=2))
    response = _client().post(
        "/v1/room/stream",
        json={
            "motion": MOTION,
            "members": [m.model_dump() for m in _cast(3)],
            "rounds": 2,
            "replicates": 1,
            "placebo_replicates": 1,
        },
    )
    assert response.status_code == 200
    frames = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [f["type"] for f in frames]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "error" not in kinds

    for frame in frames[:-1]:
        assert PRIVATE_MARKER not in json.dumps(frame)
    done = frames[-1]["result"]
    assert PRIVATE_MARKER in json.dumps(done["turns"])
    # `_sse` stamped the run id and ran the findings pass.
    assert done["run_id"]
    assert "findings" in done


# ──────────────────────────── the analysis contract ───────────────────────────


def test_analysis_payload_matches_room_analysis(monkeypatch):
    """The seam to the math module, checked by signature rather than at runtime.

    `boardroom` owns the transcript and `room` owns the estimators; if the
    keywords drift apart the failure would otherwise land at the very end of a
    paid run, after every model call has been spent. Skipped while the module is
    absent so the orchestration suite does not depend on it.
    """
    room = pytest.importorskip("app.room")
    if not hasattr(room, "room_analysis"):
        pytest.skip("app.room provides no room_analysis yet")

    import inspect

    request = _request(seats=3, rounds=2, replicates=1, placebo=1)
    payload = boardroom._analysis_payload(
        request, _cast(3), [], [[0, 1, 2], [2, 0, 1]], list(_NAMES[:3])
    )
    signature = inspect.signature(room.room_analysis)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if not accepts_kwargs:
        unknown = set(payload) - set(signature.parameters)
        assert not unknown, f"room_analysis does not accept {sorted(unknown)}"
    missing = [
        name
        for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in payload
    ]
    assert not missing, f"room_analysis requires {missing}, which boardroom does not send"

    # One entry per replicate INDEX, including skipped ones: room.py indexes
    # `meta[rep]` positionally, so a gap would silently attribute a placebo
    # replicate's order and arm to a real one.
    meta = payload["replicates_meta"]
    assert len(meta) == request.replicates + request.placebo_replicates
    assert [m["arm"] for m in meta] == ["real", "placebo"]
    assert meta[0]["decision_threshold"] is None
    assert meta[1]["order"] == ["Cyd", "Alba", "Bo"]


def test_analysis_records_carry_the_keys_room_py_reads(monkeypatch):
    """The three translations in `_analysis_payload`, checked on real records.

    Each of these fails SILENTLY on room.py's side rather than raising: a missing
    `order` falls back to the order turns happen to appear in (substituting
    arrival order for the seeded permutation, so every cascade and order-effect
    number would measure nothing), and a missing `arm` defaults to "real" (folding
    the control arm into the treatment).
    """
    seen: dict = {}
    _install(monkeypatch, FakeRoom(seats=3, rounds=2), analysis=lambda p: seen.update(p) or {})
    collect(boardroom.stream_deliberation(_request(seats=3, rounds=2, replicates=2, placebo=1)))

    assert seen["motion"] == MOTION
    assert [m["name"] for m in seen["cast"]] == list(_NAMES[:3])
    for record in seen["turns"]:
        assert record["order"] == record["speaking_index"]
        assert record["arm"] == ("placebo" if record["placebo"] else "real")
        assert record["private_position"] is not None
    assert {r["arm"] for r in seen["turns"]} == {"real", "placebo"}
    # The arm labels on the turns and in the meta must agree, since room.py will
    # take either and a disagreement would partition the run two different ways.
    for record in seen["turns"]:
        assert seen["replicates_meta"][record["replicate"]]["arm"] == record["arm"]


def test_the_real_analysis_runs_end_to_end(monkeypatch):
    """No stub on the analysis seam: the actual estimators over an actual
    transcript. This is the only test that would have caught the signature
    mismatch `_analysis_payload` exists to bridge."""
    pytest.importorskip("app.room")
    monkeypatch.setattr(boardroom, "async_client", lambda: FakeRoom(seats=4, rounds=3))
    events = collect(
        boardroom.stream_deliberation(_request(seats=4, rounds=3, replicates=2, placebo=1))
    )
    analysis = events[-1]["result"]["analysis"]
    assert isinstance(analysis, dict)
    assert analysis["names"] == list(_NAMES[:4])
    assert analysis["n_replicates"] == 2 and analysis["n_placebo_replicates"] == 1
    assert analysis["n_rounds"] == 3
    assert "placebo" in analysis and "roster" in analysis
    assert json.dumps(analysis)  # must survive the SSE encoder


# ────────────────────── the schedule: two concurrent waves ────────────────────


class _LatentRoom(FakeRoom):
    """FakeRoom with a per-turn delay, so wall clock reveals the schedule.

    Without a delay every call returns instantly and a sequential engine is
    indistinguishable from a concurrent one — the timing assertion below is the
    only thing that can tell them apart.
    """

    def __init__(self, *a, latency: float = 0.05, **kw):
        super().__init__(*a, **kw)
        self.latency = latency

    async def create(self, **kw):
        if kw["response_format"]["json_schema"]["name"] != "RoomCast":
            await asyncio.sleep(self.latency)
        return await super().create(**kw)


def test_replicates_run_concurrently_within_their_wave(monkeypatch):
    """Replicates are independent, so serialising them was pure wall clock.

    Members inside a round must still speak in order — that precedence is what
    the cascade estimate is identified on — but nothing links one replicate to
    another, and running them end-to-end cost 8.4 minutes at the default 5 seats
    x 4 rounds x (5 + 1) replicates: 150 network round-trips on a machine at 0%
    CPU. Two waves (every real replicate, then the placebos, which need a
    finished donor) put the wall clock at the slowest single replicate per wave.

    Measured here at 96 turns x 50 ms: 1.45 s against a 4.80 s sequential floor.
    The assertion is deliberately loose — CI timing is noisy — but a regression
    to a sequential loop cannot pass it, because sequential IS the floor.
    """
    seats, rounds, reps, plac, latency = 4, 3, 4, 2, 0.05
    fake = _install(monkeypatch, _LatentRoom(seats=seats, rounds=rounds, latency=latency))
    request = _request(
        seats=seats, rounds=rounds, replicates=reps, placebo=plac
    )
    calls = seats * (1 + rounds) * (reps + plac)
    sequential_floor = calls * latency

    started = time.perf_counter()
    events = collect(boardroom.stream_deliberation(request))
    elapsed = time.perf_counter() - started

    assert elapsed < sequential_floor * 0.6, (
        f"{elapsed:.2f}s against a {sequential_floor:.2f}s sequential floor — "
        "the replicates are not overlapping"
    )
    # Nothing was traded away for the speed: every slot still ran exactly once.
    result = next(e for e in events if e["type"] == "done")["result"]
    assert len(result["turns"]) == calls
    assert fake.turn_calls == calls
    assert len([e for e in events if e["type"] == "replicate_done"]) == reps + plac


def test_concurrent_waves_interleave_frames_but_keep_each_replicate_ordered(monkeypatch):
    """Frames from different replicates now arrive interleaved, and the UI
    demultiplexes on `replicate`. What must still hold is the order WITHIN a
    replicate: rounds ascending, and the sealed ballot before any deliberation."""
    fake = _install(monkeypatch, _LatentRoom(seats=3, rounds=3, latency=0.02))
    events = collect(
        boardroom.stream_deliberation(
            _request(seats=3, rounds=3, replicates=3, placebo=1)
        )
    )
    turns = [e for e in events if e["type"] == "turn"]
    assert any(
        turns[i]["replicate"] != turns[i + 1]["replicate"] for i in range(len(turns) - 1)
    ), "no interleaving — the waves did not overlap"

    for replicate in {t["replicate"] for t in turns}:
        rounds_seen = [t["round"] for t in turns if t["replicate"] == replicate]
        assert rounds_seen == sorted(rounds_seen), (replicate, rounds_seen)
        assert rounds_seen[0] == 0, "the sealed ballot must come first"


def test_a_placebo_still_only_hears_its_donor_under_concurrency(monkeypatch):
    """The control's whole mechanism, re-asserted against the new schedule.

    This is the assertion the concurrency change could most plausibly have
    broken: placebos run in a second wave precisely so their donor is finished,
    and if a placebo ever read a half-built transcript it would be hearing its
    own room and the control would be silently inert.
    """
    reps = 3
    fake = _install(monkeypatch, _LatentRoom(seats=3, rounds=2, latency=0.01))
    collect(
        boardroom.stream_deliberation(
            _request(seats=3, rounds=2, replicates=reps, placebo=2)
        )
    )
    for prompt in fake.prompts:
        replicate = prompt["replicate"]
        if replicate < reps:
            continue  # a real arm
        donor = (replicate - reps) % reps
        heard = set(re.findall(r"rep=(\d+)", str(prompt["messages"])))
        heard.discard(str(replicate))  # its own prior turns are its own words
        assert heard <= {str(donor)}, (
            f"placebo {replicate} heard {heard}, donor is {donor}"
        )

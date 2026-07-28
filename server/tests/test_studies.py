import pytest

from app.studies import aggregate_objections, aggregate_price_curves
from app.schemas import TwinObjection


def test_price_aggregation_and_optimum():
    prices = [10.0, 20.0, 40.0]
    # 3 twins; demand falls as price rises
    curves = [
        [1.0, 0.8, 0.2],
        [0.9, 0.6, 0.1],
        [1.0, 0.7, 0.3],
    ]
    result = aggregate_price_curves(curves, prices, "low", "A gadget")
    assert result.twin_count == 3
    demands = {p.price: p.demand for p in result.points}
    assert demands[10.0] > demands[40.0]  # demand decreases with price

    # revenue = price*demand: 10*0.966=9.66, 20*0.7=14, 40*0.2=8 -> peak at 20
    assert result.recommended_price == 20.0
    rev = {p.price: p.revenue_index for p in result.points}
    assert rev[20.0] == 1.0  # normalized peak
    assert rev[40.0] < 1.0


def test_price_empty_raises():
    with pytest.raises(RuntimeError):
        aggregate_price_curves([], [10.0, 20.0], "low", "x")


def _obj(sentiment, objection, dealbreaker, rec):
    return TwinObjection(
        sentiment=sentiment,
        biggest_objection=objection,
        is_dealbreaker=dealbreaker,
        would_recommend=rec,
    )


def test_objection_aggregation():
    objs = [
        _obj("negative", "Too expensive", True, 0.1),
        _obj("neutral", "Too expensive", True, 0.3),
        _obj("neutral", "Not sure it's trustworthy", False, 0.4),
        _obj("positive", "none", False, 0.9),
    ]
    result = aggregate_objections(objs, "medium", "Buy our thing")
    assert result.twin_count == 4
    assert result.sentiment_distribution["neutral"] == 2
    assert result.dealbreaker_rate == 0.5  # 2 of 4
    # "none" is filtered from themes; "Too expensive" is the top theme with 2 dealbreakers
    top = result.top_objections[0]
    assert top.objection == "Too expensive"
    assert top.count == 2
    assert top.dealbreaker_count == 2
    assert all(t.objection.lower() != "none" for t in result.top_objections)


def test_objection_empty_raises():
    with pytest.raises(RuntimeError):
        aggregate_objections([], "low", "x")

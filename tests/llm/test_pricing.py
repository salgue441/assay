import pytest

from assay.llm import TokenPrices
from assay.llm.pricing import cost_usd


def test_prices_are_per_million_tokens() -> None:
    prices = TokenPrices(input_per_million=3.0, output_per_million=15.0)

    assert prices.cost_usd(1_000, 200) == pytest.approx(0.003 + 0.003)


def test_unpriced_calls_cost_nothing_known() -> None:
    assert cost_usd(None, 1_000, 200) == 0.0
    assert cost_usd(TokenPrices(1.0, 2.0), 1_000_000, 500_000) == pytest.approx(2.0)


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        TokenPrices(input_per_million=-1.0, output_per_million=1.0)

"""
What a hosted model charges per token.
"""

from __future__ import annotations

from dataclasses import dataclass

_TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True, slots=True)
class TokenPrices:
    """
    US dollars per million input (prompt) and output tokens. Providers do not
    report prices in their responses, so they come from configuration and must
    be kept in step with the provider's price list.
    """

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        if self.input_per_million < 0 or self.output_per_million < 0:
            msg = "token prices cannot be negative"
            raise ValueError(msg)

    def cost_usd(self, prompt_tokens: int, output_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / _TOKENS_PER_MILLION


def cost_usd(prices: TokenPrices | None, prompt_tokens: int, output_tokens: int) -> float:
    """
    The cost of one call, or 0.0 when no prices are configured. A zero cost
    means the spend is unknown, so a budget cannot cap it.
    """

    return 0.0 if prices is None else prices.cost_usd(prompt_tokens, output_tokens)

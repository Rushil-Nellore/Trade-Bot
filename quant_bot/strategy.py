from __future__ import annotations


def should_exit_trade(
    *,
    price: float,
    buy_price: float,
    peak_price: float,
    trailing_stop_pct: float,
    hard_stop_pct: float,
) -> str | None:
    if price < buy_price * (1 - hard_stop_pct):
        return "hard_stop"

    drop_from_peak = (peak_price - price) / peak_price
    if drop_from_peak >= trailing_stop_pct:
        return "trailing_stop"

    return None


from __future__ import annotations  # enables newer type-hint syntax on older Python versions without breaking anything


def should_exit_trade(  # define a function that decides whether to close an open trade
    *,  # the bare * forces every argument to be passed by name (e.g. price=95.0) — prevents passing them in the wrong order by accident
    price: float,  # the current BTC price at this candle
    buy_price: float,  # the price at which we originally entered (bought into) this trade
    peak_price: float,  # the highest price BTC reached at any point since we bought — updated each candle
    trailing_stop_pct: float,  # how far price must fall from its peak before we sell to lock in profits (e.g. 0.05 = 5%)
    hard_stop_pct: float,  # how far price must fall below our entry price before we emergency-exit to limit losses (e.g. 0.04 = 4%)
) -> str | None:  # return type: either a string naming the exit reason, or None meaning "stay in the trade"
    if price < buy_price * (1 - hard_stop_pct):  # check hard stop: multiply entry price by 0.96 — if current price is below that floor, we've lost 4% and must exit
        return "hard_stop"  # return this string to tell the caller which exit rule triggered

    drop_from_peak = (peak_price - price) / peak_price  # calculate how far price has fallen from its highest point as a fraction; e.g. peak=$110k, price=$104k → (110-104)/110 = 5.45%
    if drop_from_peak >= trailing_stop_pct:  # if the drop from peak meets or exceeds the threshold (5%), the trailing stop fires
        return "trailing_stop"  # return this string so the caller knows it was a trailing stop, not a hard stop

    return None  # neither stop condition was met — return None to signal "keep holding the trade"

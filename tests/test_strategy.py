import unittest

from quant_bot.strategy import should_exit_trade


class StrategyTests(unittest.TestCase):

    def test_returns_hard_stop_before_trailing_stop(self) -> None:
        reason = should_exit_trade(
            price=95.0,
            buy_price=100.0,
            peak_price=110.0,
            trailing_stop_pct=0.05,
            hard_stop_pct=0.04,
        )
        self.assertEqual(reason, "hard_stop")

    def test_returns_trailing_stop_after_peak_drop(self) -> None:
        reason = should_exit_trade(
            price=104.0,
            buy_price=100.0,
            peak_price=110.0,
            trailing_stop_pct=0.05,
            hard_stop_pct=0.04,
        )
        self.assertEqual(reason, "trailing_stop")

    def test_returns_none_when_exit_not_triggered(self) -> None:
        reason = should_exit_trade(
            price=108.0,
            buy_price=100.0,
            peak_price=110.0,
            trailing_stop_pct=0.05,
            hard_stop_pct=0.04,
        )
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()

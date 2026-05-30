import unittest  # Python's built-in testing framework — provides TestCase base class and assertion methods like assertEqual, assertIsNone

from quant_bot.strategy import should_exit_trade  # import the function we want to test


class StrategyTests(unittest.TestCase):  # define a test class that inherits from TestCase — unittest automatically discovers and runs all methods starting with "test_"

    def test_returns_hard_stop_before_trailing_stop(self) -> None:  # test: when BOTH stops would trigger, the hard stop should take priority
        reason = should_exit_trade(  # call the function with values where price ($95) is below the hard-stop floor ($100 × 0.96 = $96) AND below the trailing-stop floor ($110 × 0.95 = $104.50)
            price=95.0,          # current price — 5% below entry, triggers hard stop (4% threshold)
            buy_price=100.0,     # entry price
            peak_price=110.0,    # highest price since entry
            trailing_stop_pct=0.05,  # trailing stop fires if price drops 5% from peak
            hard_stop_pct=0.04,      # hard stop fires if price drops 4% below entry
        )
        self.assertEqual(reason, "hard_stop")  # assertEqual(): assert the returned value exactly equals "hard_stop" — if not, the test fails with a descriptive error

    def test_returns_trailing_stop_after_peak_drop(self) -> None:  # test: trailing stop fires when price drops from its peak but NOT below the hard-stop floor
        reason = should_exit_trade(
            price=104.0,         # $104 is above $100×0.96=$96 (no hard stop) but below $110×0.95=$104.50 (trailing stop triggers)
            buy_price=100.0,
            peak_price=110.0,
            trailing_stop_pct=0.05,
            hard_stop_pct=0.04,
        )
        self.assertEqual(reason, "trailing_stop")  # verify the trailing stop string is returned

    def test_returns_none_when_exit_not_triggered(self) -> None:  # test: when price is comfortably above both stop levels, no exit should trigger
        reason = should_exit_trade(
            price=108.0,         # $108 is above hard-stop floor ($96) and only 1.8% below peak ($110) — well below the 5% trailing threshold
            buy_price=100.0,
            peak_price=110.0,
            trailing_stop_pct=0.05,
            hard_stop_pct=0.04,
        )
        self.assertIsNone(reason)  # assertIsNone(): assert the result is exactly None — confirms no exit was triggered


if __name__ == "__main__":  # allow running this file directly (python tests/test_strategy.py) as well as via python -m unittest
    unittest.main()  # unittest.main(): discovers and runs all test methods in this file

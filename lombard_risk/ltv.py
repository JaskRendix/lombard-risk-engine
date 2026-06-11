from dataclasses import dataclass


@dataclass
class LTVProfile:
    portfolio_value: float
    loan_amount: float
    haircut: float

    @property
    def max_ltv(self) -> float:
        return 1.0 - self.haircut

    @property
    def current_ltv(self) -> float:
        return self.loan_amount / self.portfolio_value

    def margin_call_triggered(self, threshold: float) -> bool:
        return self.current_ltv > threshold

    @property
    def stress_ltv(self) -> float:
        """
        LTV after haircut is applied to collateral value.
        Pure math: loan / (portfolio * (1 - haircut)).
        """
        stressed_value = self.portfolio_value * (1.0 - self.haircut)
        return self.loan_amount / stressed_value

    @property
    def post_haircut_value(self) -> float:
        """
        Collateral value after haircut.
        Pure math: portfolio * (1 - haircut).
        """
        return self.portfolio_value * (1.0 - self.haircut)

    def breach_distance(self, threshold: float) -> float:
        """
        Distance to margin-call threshold.
        Positive = safe buffer.
        Negative = breached.
        """
        return threshold - self.current_ltv

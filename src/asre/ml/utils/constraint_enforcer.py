"""Constraint Enforcer"""
import pandas as pd

class ConstraintEnforcer:
    @staticmethod
    def enforce_ordering(thresholds: pd.DataFrame) -> pd.DataFrame:
        # TODO: Ensure exit < reduce < full < overweight
        return thresholds

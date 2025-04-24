from enum import Enum
import sys

from koozie import fr_u


# Defrost characterisitcs
class DefrostControl(Enum):
    NONE = 0
    TIMED = 1
    DEMAND = 2


class DefrostStrategy(Enum):
    NONE = 0
    REVERSE_CYCLE = 1
    RESISTIVE = 2


class Defrost:
    def __init__(
        self,
        time_fraction=lambda conditions: fr_u(3.5, "min") / fr_u(60.0, "min"),
        resistive_power=0,
        control=DefrostControl.TIMED,
        strategy=DefrostStrategy.REVERSE_CYCLE,
        high_temperature=fr_u(40, "°F"),
        low_temperature=None,  # Minimum temperature for defrost operation
        period=fr_u(90, "min"),  # Time between defrost terminations (for testing)
        max_time=fr_u(720, "min"),
    ):  # Maximum time between defrosts allowed by controls
        # Initialize member values
        self.time_fraction = time_fraction
        self.resistive_power = resistive_power
        self.control = control
        self.strategy = strategy
        self.high_temperature = high_temperature
        self.low_temperature = low_temperature
        self.period = period
        self.max_time = max_time

        # Check inputs
        if self.strategy == DefrostStrategy.RESISTIVE and self.resistive_power <= 0:
            sys.exit(f"Defrost strategy=RESISTIVE, but resistive_power is not greater than zero.")

    def in_defrost(self, conditions):
        if self.strategy == DefrostStrategy.NONE:
            return False
        if self.low_temperature is not None:
            if conditions.outdoor.db > self.low_temperature and conditions.outdoor.db < self.high_temperature:
                return True
        else:
            if conditions.outdoor.db < self.high_temperature:
                return True
        return False

    def demand_credit(self) -> float:
        """Defrost credit as defined by AHRI 210/240."""
        if self.control == DefrostControl.DEMAND:
            t_test = max(self.period, fr_u(90, "min"))
            t_max = min(self.max_time, fr_u(720.0, "min"))

            return 1.0 + 0.03 * (1 - (t_test - fr_u(90.0, "min")) / (t_max - fr_u(90.0, "min")))  # eq. 11.129
        return 1.0  # eq. 11.130

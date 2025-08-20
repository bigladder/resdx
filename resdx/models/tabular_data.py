from bisect import insort
from copy import deepcopy

from koozie import fr_u, to_u
from scipy.interpolate import RegularGridInterpolator

from ..enums import StagingType
from ..util import bracket, calc_biquad
from .nrel import NRELDXModel
from .rating_correlations import cop_47_h1_full, cop_82_b_low
from .statistical_set import StatisticalSet, original_statistics


class TemperatureSpeedPerformanceTable:
    def __init__(
        self,
        temperatures: list[float],
        number_of_speeds: int,
        data_in: list[list[float | None]] | None = None,
    ) -> None:
        self.temperatures: list[float] = deepcopy(temperatures)
        self.speeds: list[int] = [i + 1 for i in range(number_of_speeds)]
        self.data: list[list[float | None]]
        if data_in is None:
            self.data = [[None] * number_of_speeds for _ in range(len(temperatures))]
        else:
            self.data = data_in

        self.interpolator: RegularGridInterpolator

    def set(self, speed: int, temperature: float, value: float) -> None:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        self.data[temperature_index][speed_index] = value

    def get(self, speed: int, temperature: float) -> float | None:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        return self.data[temperature_index][speed_index]

    def set_by_ratio(self, speed: int, temperature: float, ratio: float) -> None:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        max_index = len(self.speeds) - 1
        if speed_index == max_index:
            # Set max from rated
            reference_speed_index = speed_index - 1
            self.data[temperature_index][speed_index] = self.data[temperature_index][reference_speed_index] / ratio
        else:
            # Set from max
            self.data[temperature_index][speed_index] = self.data[temperature_index][max_index] * ratio

    def get_ratio(self, speed: int, temperature: float) -> float:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        max_index = len(self.speeds) - 1
        return self.data[temperature_index][speed_index] / self.data[temperature_index][max_index]

    def set_by_interpolation(self, speed: int, temperature: float) -> None:
        """Interpolate rated speed from temperature above (currently only needed for lower temperatures)"""
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        s1 = speed_index
        s0 = s1 - 1
        s2 = s1 + 1
        t1 = temperature_index
        t2 = t1 + 1
        self.data[t1][s1] = self.data[t1][s0] + (self.data[t2][s1] - self.data[t2][s0]) / (
            self.data[t2][s2] - self.data[t2][s0]
        ) * (self.data[t1][s2] - self.data[t1][s0])

    def get_temperature_index(self, temperature: float) -> int:
        if temperature in self.temperatures:
            return self.temperatures.index(temperature)
        else:
            raise RuntimeError(f"Temperature, {temperature:.2f}, not found.")

    def add_temperature(
        self,
        temperature: float,
        extrapolate: bool = True,
        extrapolation_limit: float | None = None,
    ) -> None:
        if temperature in self.temperatures:
            raise RuntimeError(f"Temperature, {temperature:.2f}, already exists. Unable to add temperature.")
        insort(self.temperatures, temperature)
        t_i = self.get_temperature_index(temperature)  # temperature index

        self.data.insert(t_i, [None] * len(self.speeds))

        if t_i == 0:
            # Extrapolate below
            t_0 = temperature
            t_1 = self.temperatures[1]
            t_2 = self.temperatures[2]
            for speed in self.speeds:
                s_i = speed - 1  # speed index
                if extrapolate:
                    value = self.data[t_i + 1][s_i] - (self.data[t_i + 2][s_i] - self.data[t_i + 1][s_i]) / (
                        t_2 - t_1
                    ) * (t_1 - t_0)
                    if extrapolation_limit is not None:
                        value = max(value, self.data[t_i + 1][s_i] * extrapolation_limit)
                else:
                    value = self.data[t_i + 1][s_i]

                self.data[t_i][s_i] = value

        elif t_i == len(self.temperatures) - 1:
            # Extrapolate above
            t = temperature
            t_m1 = self.temperatures[t_i - 1]
            t_m2 = self.temperatures[t_i - 2]
            for speed in self.speeds:
                s_i = speed - 1  # speed index
                if extrapolate:
                    value = self.data[t_i - 1][s_i] + (self.data[t_i - 1][s_i] - self.data[t_i - 2][s_i]) / (
                        t_m1 - t_m2
                    ) * (t - t_m1)
                    if extrapolation_limit is not None:
                        value = min(value, self.data[t_i - 1][s_i] * extrapolation_limit)
                else:
                    value = self.data[t_i - 1][s_i]

                self.data[t_i][s_i] = value

        else:
            # Interpolate
            t = temperature
            t_m1 = self.temperatures[t_i - 1]
            t_1 = self.temperatures[1]
            for speed in self.speeds:
                s_i = speed - 1  # speed index
                self.data[t_i][s_i] = self.data[t_i - 1][s_i] + (self.data[t_i + 1][s_i] - self.data[t_i - 1][s_i]) / (
                    t_1 - t_m1
                ) * (t - t_m1)

    def get_temperature_at_value(
        self,
        value: float,
        temperature_1: float,
        temperature_2: float,
        speed: int,
    ) -> float:
        """Using a line between two points for a given speed, calculate the temperature corresponding to a given value"""
        v_1 = self.data[self.get_temperature_index(temperature_1)][speed - 1]
        v_2 = self.data[self.get_temperature_index(temperature_2)][speed - 1]
        assert v_1 is not None and v_2 is not None
        return temperature_1 + (value - v_1) / (v_2 - v_1) * (temperature_2 - temperature_1)

    def set_by_maintenance(self, speed: int, temperature: float, reference_temperature: float, ratio: float) -> None:
        raise NotImplementedError()

    def get_maintenance(self, speed: int, temperature: float, reference_temperature: float) -> float:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        reference_temperature_index = self.get_temperature_index(reference_temperature)
        value = self.data[temperature_index][speed_index]
        reference_value = self.data[reference_temperature_index][speed_index]
        assert value is not None and reference_value is not None
        return value / reference_value

    def set_interpolator(self):
        self.interpolator = RegularGridInterpolator(
            points=(self.temperatures, self.speeds),
            values=self.data,
            method="linear",
            bounds_error=False,  # Do not raise an error if the value is out of bounds
            fill_value=None,  # Fill by extrapolation
        )

    def calculate(self, speed: float, temperature: float) -> float:
        return self.interpolator([temperature, speed])[0]

    def apply_fan_power_correction(self, fan_powers: list[float]) -> None:
        assert len(fan_powers) == len(self.speeds)
        for t_i, speed_data in enumerate(self.data):
            for s_i, value in enumerate(speed_data):
                self.data[t_i][s_i] = value - fan_powers[s_i]
                if self.data[t_i][s_i] <= 0:
                    raise RuntimeError(
                        f"Negative value after fan power correction. Value={value:.2f} W, Fan Power={fan_powers[s_i]:.2f} W at temperature={to_u(self.temperatures[t_i], '°F'):.1f} °F, speed={self.speeds[s_i]}"
                    )

        self.set_interpolator()


class TemperatureSpeedCoolingPerformanceTable(TemperatureSpeedPerformanceTable):
    def set_by_maintenance(self, speed: int, temperature: float, reference_temperature: float, ratio: float) -> None:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        reference_temperature_index = self.get_temperature_index(reference_temperature)
        if reference_temperature_index < temperature_index:
            self.data[temperature_index][speed_index] = self.data[reference_temperature_index][speed_index] * ratio
        else:
            self.data[temperature_index][speed_index] = self.data[reference_temperature_index][speed_index] / ratio


class TemperatureSpeedHeatingPerformanceTable(TemperatureSpeedPerformanceTable):
    def set_by_maintenance(self, speed: int, temperature: float, reference_temperature: float, ratio: float) -> None:
        speed_index = speed - 1
        temperature_index = self.get_temperature_index(temperature)
        reference_temperature_index = self.get_temperature_index(reference_temperature)
        if reference_temperature_index > temperature_index:
            self.data[temperature_index][speed_index] = self.data[reference_temperature_index][speed_index] * ratio
        else:
            self.data[temperature_index][speed_index] = self.data[reference_temperature_index][speed_index] / ratio


class TemperatureSpeedPerformance:
    def __init__(
        self,
        cooling_capacities: TemperatureSpeedCoolingPerformanceTable,
        cooling_powers: TemperatureSpeedCoolingPerformanceTable,
        heating_capacities: TemperatureSpeedHeatingPerformanceTable,
        heating_powers: TemperatureSpeedHeatingPerformanceTable,
    ) -> None:
        self.number_of_cooling_speeds = len(cooling_capacities.speeds)
        if len(cooling_powers.speeds) != self.number_of_cooling_speeds:
            raise RuntimeError("Inconsistent power / capacity table sizes!")
        self.number_of_cooling_temperatures = len(cooling_capacities.temperatures)
        if len(cooling_powers.temperatures) != self.number_of_cooling_temperatures:
            raise RuntimeError("Inconsistent power / capacity table sizes!")

        self.number_of_heating_speeds = len(heating_capacities.speeds)
        if len(heating_powers.speeds) != self.number_of_heating_speeds:
            raise RuntimeError("Inconsistent power / capacity table sizes!")
        self.number_of_heating_temperatures = len(heating_capacities.temperatures)
        if len(heating_powers.temperatures) != self.number_of_heating_temperatures:
            raise RuntimeError("Inconsistent power / capacity table sizes!")

        self.cooling_capacities = cooling_capacities
        self.cooling_powers = cooling_powers
        self.heating_capacities = heating_capacities
        self.heating_powers = heating_powers
        self.rated_cooling_speed = 2
        self.rated_heating_speed = 2

    def cooling_capacity(self, speed: float = 2, temperature: float = fr_u(95, "degF")) -> float:
        return self.cooling_capacities.calculate(speed, temperature)

    def cooling_power(self, speed: float = 2, temperature: float = fr_u(95, "degF")) -> float:
        return self.cooling_powers.calculate(speed, temperature)

    def cooling_cop(self, speed: float = 2, temperature: float = fr_u(95, "degF")) -> float:
        return self.cooling_capacity(speed, temperature) / self.cooling_power(speed, temperature)

    def heating_capacity(self, speed: float = 2, temperature: float = fr_u(47, "degF")) -> float:
        return self.heating_capacities.calculate(speed, temperature)

    def heating_power(self, speed: float = 2, temperature: float = fr_u(47, "degF")) -> float:
        return self.heating_powers.calculate(speed, temperature)

    def heating_cop(self, speed: float = 2, temperature: float = fr_u(47, "degF")) -> float:
        return self.heating_capacity(speed, temperature) / self.heating_power(speed, temperature)

    def make_gross(self, cooling_fan_powers: list[float], heating_fan_powers: list[float]) -> None:
        self.cooling_capacities.apply_fan_power_correction([-p for p in cooling_fan_powers])  # increases
        self.cooling_powers.apply_fan_power_correction(cooling_fan_powers)  # decreases
        self.heating_capacities.apply_fan_power_correction(heating_fan_powers)  # decreases
        self.heating_powers.apply_fan_power_correction(heating_fan_powers)  # decreases


def make_neep_statistical_model_data(
    cooling_capacity_95: float,  # Net total cooling capacity at 95F and rated speed
    eer2: float,
    heating_capacity_47: float,
    heating_capacity_17: float | None = None,
    seer2: float | None = None,
    hspf2: float | None = None,
    min_heating_temperature: float = fr_u(-20, "degF"),
    cooling_capacity_ratio: float | None = None,  # min/max capacity ratio at 95F
    cooling_cop_82_min: float | None = None,
    heating_cop_47: float | None = None,
    default_heating_statistics: StatisticalSet = original_statistics,
    default_cooling_statistics: StatisticalSet | None = None,
) -> TemperatureSpeedPerformance:
    if default_cooling_statistics is None:
        default_cooling_statistics = default_heating_statistics

    Qmin = 1
    Qrated = 2
    Qmax = 3

    # COOLING

    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")

    t_c = [
        t_82,
        t_95,
    ]

    Qr95rated = default_cooling_statistics.Qr95Rated
    Qm95max = default_cooling_statistics.Qm95Max
    Qm95min = default_cooling_statistics.Qm95Min
    EIRr95rated = default_cooling_statistics.EIRr95Rated
    EIRm95max = default_cooling_statistics.EIRm95Max
    EIRm95min = default_cooling_statistics.EIRm95Min

    Q_c = TemperatureSpeedCoolingPerformanceTable(t_c, 3)
    P_c = TemperatureSpeedCoolingPerformanceTable(t_c, 3)

    # Net Total Capacity

    # 95 F
    Q_c.set(Qrated, t_95, cooling_capacity_95)
    Q_c.set_by_ratio(Qmax, t_95, Qr95rated)

    # 82 F
    Q_c.set_by_maintenance(Qmax, t_82, t_95, Qm95max)
    # Other speeds calculated later

    # Net Power
    Pr95rated = Qr95rated * EIRr95rated
    Pm95min = Qm95min * EIRm95min
    Pm95max = Qm95max * EIRm95max

    # 95/82 F
    P_c.set(Qrated, t_95, Q_c.get(Qrated, t_95) / fr_u(eer2, "Btu/Wh"))
    P_c.set_by_ratio(Qmax, t_95, Pr95rated)
    P_c.set_by_maintenance(Qmax, t_82, t_95, Pm95max)
    if cooling_cop_82_min is None:
        if seer2 is None:
            raise ValueError("seer2 cannot be None.")
        cooling_cop_82_min = cop_82_b_low(StagingType.VARIABLE_SPEED, seer2, seer2 / eer2)

    EIRr82min = (Q_c.get(Qmax, t_82) / P_c.get(Qmax, t_82)) / cooling_cop_82_min

    if cooling_capacity_ratio is not None:
        Qr95min = cooling_capacity_ratio
    else:
        m1 = default_cooling_statistics.EIRr82MinvSEER2_over_EER2.slope
        b1 = default_cooling_statistics.EIRr82MinvSEER2_over_EER2.intercept
        m2 = default_cooling_statistics.Qr95MinvSEER2_over_EER2.slope
        b2 = default_cooling_statistics.Qr95MinvSEER2_over_EER2.intercept
        # Algebraicly solve for Qr95min as a function of EIRr82min in case COP82min is explcicitly set
        slope = m2 / m1
        intercept = b2 - slope * b1
        Qr95min = bracket(intercept + slope * EIRr82min, 0.1, Qr95rated)

    # Back to capacities
    Q_c.set_by_ratio(Qmin, t_95, Qr95min)
    Q_c.set_by_maintenance(Qmin, t_82, t_95, Qm95min)
    Q_c.set_by_interpolation(Qrated, t_82)

    P_c.set(Qmin, t_82, Q_c.get(Qmin, t_82) / cooling_cop_82_min)

    P_c.set_by_maintenance(Qmin, t_95, t_82, Pm95min)
    P_c.set_by_interpolation(Qrated, t_82)

    # Tmin
    # Find temperature where power is half of 82F value to avoid division by zero

    t_c_min = P_c.get_temperature_at_value(P_c.get(Qmin, t_82) * 0.5, t_82, t_95, Qmin)

    Q_c.add_temperature(t_c_min)
    P_c.add_temperature(t_c_min)

    t_40 = fr_u(40.0, "degF")
    if t_c_min > t_40:
        Q_c.add_temperature(t_40, extrapolate=True)
        P_c.add_temperature(t_40, extrapolate=False)

    # HEATING

    t_h_min = min_heating_temperature
    t_5 = fr_u(5, "degF")
    t_17 = fr_u(17, "degF")
    t_47 = fr_u(47, "degF")

    t_h = [
        t_h_min,
        t_5,
        t_17,
        t_47,
    ]

    Qr47rated = default_heating_statistics.Qr47Rated
    Qr47min = default_heating_statistics.Qr47Min
    Qr17rated = default_heating_statistics.Qr17Rated
    Qr17min = default_heating_statistics.Qr17Min
    Qm5max = default_heating_statistics.Qm5Max
    Qr5rated = default_heating_statistics.Qr5Rated
    Qr5min = default_heating_statistics.Qr5Min
    QmslopeLCTmax = default_heating_statistics.QmslopeLCTMax
    QmslopeLCTmin = default_heating_statistics.QmslopeLCTMin
    EIRr47rated = default_heating_statistics.EIRr47Rated
    EIRr47min = default_heating_statistics.EIRr47Min
    EIRm17rated = default_heating_statistics.EIRm17Rated
    EIRr17rated = default_heating_statistics.EIRr17Rated
    EIRr17min = default_heating_statistics.EIRr17Min
    EIRm5max = default_heating_statistics.EIRm5Max
    EIRr5rated = default_heating_statistics.EIRr5Rated
    EIRr5min = default_heating_statistics.EIRr5Min
    EIRmslopeLCTmax = default_heating_statistics.EIRmslopeLCTMax
    EIRmslopeLCTmin = default_heating_statistics.EIRmslopeLCTMin

    if heating_capacity_17 is not None:
        Qm17rated = heating_capacity_17 / heating_capacity_47
    else:
        Qm17rated = default_heating_statistics.Qm17Rated

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 3)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 3)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)
    Q_h.set_by_ratio(Qmax, t_47, Qr47rated)
    Q_h.set_by_ratio(Qmin, t_47, Qr47min)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)
    Q_h.set_by_ratio(Qmax, t_17, Qr17rated)
    Q_h.set_by_ratio(Qmin, t_17, Qr17min)

    # 5 F
    Q_h.set_by_maintenance(Qmax, t_5, t_17, Qm5max)
    Q_h.set_by_ratio(Qrated, t_5, Qr5rated)
    Q_h.set_by_ratio(Qmin, t_5, Qr5min)

    QmLCTmax = 1 / (1 - fr_u(QmslopeLCTmax, "1/degF") * (t_5 - t_h_min))
    QmLCTmin = 1 / (1 - fr_u(QmslopeLCTmin, "1/degF") * (t_5 - t_h_min))

    # Tmin
    Q_h.set_by_maintenance(Qmax, t_h_min, t_5, QmLCTmax)
    Q_h.set_by_maintenance(Qmin, t_h_min, t_5, QmLCTmin)
    Q_h.set_by_interpolation(Qrated, t_h_min)

    # Net Power
    if heating_cop_47 is None:
        if hspf2 is None:
            raise ValueError("hspf2 cannot be None")
        heating_cop_47 = cop_47_h1_full(StagingType.VARIABLE_SPEED, hspf2, Qm17rated)

    Pr47rated = Qr47rated * EIRr47rated
    Pr47min = Qr47min * EIRr47min
    Pm17rated = Qm17rated * EIRm17rated
    Pr17rated = Qr17rated * EIRr17rated
    Pr17min = Qr17min * EIRr17min
    Pm5max = Qm5max * EIRm5max
    Pr5rated = Qr5rated * EIRr5rated
    Pr5min = Qr5min * EIRr5min

    # 47 F
    P_h.set(Qrated, t_47, Q_h.get(Qrated, t_47) / heating_cop_47)
    P_h.set_by_ratio(Qmax, t_47, Pr47rated)
    P_h.set_by_ratio(Qmin, t_47, Pr47min)

    # 17 F
    P_h.set_by_maintenance(Qrated, t_17, t_47, Pm17rated)
    P_h.set_by_ratio(Qmax, t_17, Pr17rated)
    P_h.set_by_ratio(Qmin, t_17, Pr17min)

    # 5 F
    P_h.set_by_maintenance(Qmax, t_5, t_17, Pm5max)
    P_h.set_by_ratio(Qrated, t_5, Pr5rated)
    P_h.set_by_ratio(Qmin, t_5, Pr5min)

    EIRmLCTmax = 1 / (1 - fr_u(EIRmslopeLCTmax, "1/degF") * (t_5 - t_h_min))
    EIRmLCTmin = 1 / (1 - fr_u(EIRmslopeLCTmin, "1/degF") * (t_5 - t_h_min))

    PmLCTmax = EIRmLCTmax * Q_h.get_maintenance(Qmax, t_h_min, t_5)
    PmLCTmin = EIRmLCTmin * Q_h.get_maintenance(Qmin, t_h_min, t_5)

    # Tmin
    P_h.set_by_maintenance(Qmax, t_h_min, t_5, PmLCTmax)
    P_h.set_by_maintenance(Qmin, t_h_min, t_5, PmLCTmin)
    P_h.set_by_interpolation(Qrated, t_h_min)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)


def make_neep_model_data(
    cooling_capacities: list[list[float | None]],
    cooling_powers: list[list[float | None]],
    heating_capacities: list[list[float | None]],
    heating_powers: list[list[float | None]],
    lct: float | None = None,
) -> TemperatureSpeedPerformance:
    # Convert from NEEP units
    for s_i, value_list in enumerate(cooling_capacities):
        for t_i, value in enumerate(value_list):
            if value is not None:
                cooling_capacities[s_i][t_i] = fr_u(value, "Btu/h")

    for s_i, value_list in enumerate(cooling_powers):
        for t_i, value in enumerate(value_list):
            if value is not None:
                cooling_powers[s_i][t_i] = fr_u(value, "kW")

    for s_i, value_list in enumerate(heating_capacities):
        for t_i, value in enumerate(value_list):
            if value is not None:
                heating_capacities[s_i][t_i] = fr_u(value, "Btu/h")

    for s_i, value_list in enumerate(heating_powers):
        for t_i, value in enumerate(value_list):
            if value is not None:
                heating_powers[s_i][t_i] = fr_u(value, "kW")

    # Set up temperatures
    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")
    cooling_temperatures = [t_82, t_95]

    t_5 = fr_u(5.0, "degF")
    t_17 = fr_u(17.0, "degF")
    t_47 = fr_u(47.0, "degF")
    heating_temperatures = [t_5, t_17, t_47]
    if lct is not None:
        t_lct = fr_u(lct, "degF")
        heating_temperatures = [t_lct] + heating_temperatures

    Q_c = TemperatureSpeedCoolingPerformanceTable(cooling_temperatures, 3, cooling_capacities)
    P_c = TemperatureSpeedCoolingPerformanceTable(cooling_temperatures, 3, cooling_powers)
    Q_h = TemperatureSpeedHeatingPerformanceTable(heating_temperatures, 3, heating_capacities)
    P_h = TemperatureSpeedHeatingPerformanceTable(heating_temperatures, 3, heating_powers)

    # Interpolate for missing rated conditions, and extrapolate to extreme temperatures
    Q_c.set_by_interpolation(2, t_82)
    P_c.set_by_interpolation(2, t_82)

    # Tmin
    # Find temperature where power is half of 82F value to avoide division by zero

    t_c_min = P_c.get_temperature_at_value(P_c.get(1, t_82) * 0.5, t_82, t_95, 1)

    Q_c.add_temperature(t_c_min)
    P_c.add_temperature(t_c_min)

    t_40 = fr_u(40.0, "degF")
    if t_c_min > t_40:
        Q_c.add_temperature(t_40, extrapolate=True)
        P_c.add_temperature(t_40, extrapolate=False)

    if Q_h.get(2, t_5) is None:
        Q_h.set_by_interpolation(2, t_5)

    if P_h.get(2, t_5) is None:
        P_h.set_by_interpolation(2, t_5)

    if lct is not None:
        Q_h.set_by_interpolation(2, t_lct)
        P_h.set_by_interpolation(2, t_lct)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)


def make_single_speed_model_data(
    cooling_capacity_95: float,  # Net total cooling capacity at 95F and rated speed
    seer2: float,
    eer2: float,
    heating_capacity_47: float,
    heating_capacity_17: float | None,
    hspf2: float,
    heating_cop_47: float | None = None,
    cycling_degradation_coefficient: float = 0.08,
) -> TemperatureSpeedPerformance:
    Qrated = 1

    # COOLING

    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")

    t_c = [
        t_82,
        t_95,
    ]

    Qm95rated = 1.0 / calc_biquad(NRELDXModel.COOLING_CAP_FT_COEFFICIENTS, 67.0, 82.0)

    Q_c = TemperatureSpeedCoolingPerformanceTable(t_c, 1)
    P_c = TemperatureSpeedCoolingPerformanceTable(t_c, 1)

    # Net Total Capacity

    # 95 F
    Q_c.set(Qrated, t_95, cooling_capacity_95)

    # 82 F
    Q_c.set_by_maintenance(Qrated, t_82, t_95, Qm95rated)

    # Net Power

    # 95/82 F
    P_c.set(Qrated, t_95, Q_c.get(Qrated, t_95) / fr_u(eer2, "Btu/Wh"))
    eer2_b = seer2 / (1.0 - 0.5 * cycling_degradation_coefficient)  # EER2 at B (82F) conditions
    P_c.set(Qrated, t_82, Q_c.get(Qrated, t_82) / fr_u(eer2_b, "Btu/Wh"))

    # Tmin
    # Find temperature where power is half of 82F value to avoide division by zero

    t_c_min = P_c.get_temperature_at_value(P_c.get(1, t_82) * 0.5, t_82, t_95, 1)

    Q_c.add_temperature(t_c_min)
    P_c.add_temperature(t_c_min)

    t_40 = fr_u(40.0, "degF")
    if t_c_min > t_40:
        Q_c.add_temperature(t_40, extrapolate=True)
        P_c.add_temperature(t_40, extrapolate=False)

    # HEATING

    t_17 = fr_u(17, "degF")
    t_47 = fr_u(47, "degF")

    t_h = [
        t_17,
        t_47,
    ]

    EIRm17rated = calc_biquad(NRELDXModel.HEATING_EIR_FT_COEFFICIENTS, 70.0, 17.0)

    if heating_capacity_17 is not None:
        Qm17rated = heating_capacity_17 / heating_capacity_47
    else:
        Qm17rated = 0.626  # Based on AHRI directory units believed to be single speed (4/4/24)

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)

    # Net Power
    if heating_cop_47 is None:
        heating_cop_47 = cop_47_h1_full(StagingType.SINGLE_STAGE, hspf2, Qm17rated)

    Pm17rated = Qm17rated * EIRm17rated

    # 47 F
    P_h.set(Qrated, t_47, Q_h.get(Qrated, t_47) / heating_cop_47)

    # 17 F
    P_h.set_by_maintenance(Qrated, t_17, t_47, Pm17rated)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)


def make_two_speed_model_data(
    cooling_capacity_95: float,  # Net total cooling capacity at 95F and rated speed
    seer2: float,
    eer2: float,
    heating_capacity_47: float,
    heating_capacity_17: float | None,
    hspf2: float,
    cooling_cop_82_min: float | None = None,
    heating_cop_47: float | None = None,
) -> TemperatureSpeedPerformance:
    Qmin = 1
    Qrated = 2

    # COOLING

    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")

    t_c = [
        t_82,
        t_95,
    ]

    Qm95rated = 1.0 / calc_biquad(NRELDXModel.COOLING_CAP_FT_COEFFICIENTS, 67.0, 82.0)
    EIRm95rated = 1.0 / calc_biquad(NRELDXModel.COOLING_EIR_FT_COEFFICIENTS, 67.0, 82.0)

    QrCmin = 0.728  # Converted from gross 0.72
    # EIRrCmin = 0.869  # Converted from gross 0.91 (Not used. Kept for completeness.)

    Q_c = TemperatureSpeedCoolingPerformanceTable(t_c, 2)
    P_c = TemperatureSpeedCoolingPerformanceTable(t_c, 2)

    # Net Total Capacity

    # 95 F
    Q_c.set(Qrated, t_95, cooling_capacity_95)
    Q_c.set_by_ratio(Qmin, t_95, QrCmin)

    # 82 F
    Q_c.set_by_maintenance(Qrated, t_82, t_95, Qm95rated)
    Q_c.set_by_ratio(Qmin, t_82, QrCmin)

    # Net Power
    Pm95rated = Qm95rated * EIRm95rated

    if cooling_cop_82_min is None:
        cooling_cop_82_min = cop_82_b_low(StagingType.VARIABLE_SPEED, seer2, seer2 / eer2)

    # 82 / 95 F
    P_c.set(Qrated, t_95, Q_c.get(Qrated, t_95) / fr_u(eer2, "Btu/Wh"))
    P_c.set_by_maintenance(Qrated, t_82, t_95, Pm95rated)
    P_c.set(Qmin, t_82, Q_c.get(Qmin, t_82) / cooling_cop_82_min)
    P_c.set_by_maintenance(Qmin, t_95, t_82, Pm95rated)

    # Tmin
    # Find temperature where power is half of 82F value to avoide division by zero

    t_c_min = P_c.get_temperature_at_value(P_c.get(1, t_82) * 0.5, t_82, t_95, 1)
    Q_c.add_temperature(t_c_min)
    P_c.add_temperature(t_c_min)

    t_40 = fr_u(40.0, "degF")
    if t_c_min > t_40:
        Q_c.add_temperature(t_40, extrapolate=True)
        P_c.add_temperature(t_40, extrapolate=False)

    # HEATING

    t_17 = fr_u(17, "degF")
    t_47 = fr_u(47, "degF")

    t_h = [
        t_17,
        t_47,
    ]

    QrHmin = 0.712  # Converted from gross 0.72
    EIRrHmin = 0.850  # Converted from gross 0.87
    EIRm17rated = calc_biquad(NRELDXModel.HEATING_EIR_FT_COEFFICIENTS, 70.0, 17.0)

    if heating_capacity_17 is not None:
        Qm17rated = heating_capacity_17 / heating_capacity_47
    else:
        Qm17rated = (
            0.626  # Based on AHRI directory units believed to be single speed (4/4/24) TODO: Switch to Cutler curve
        )

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 2)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 2)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)
    Q_h.set_by_ratio(Qmin, t_47, QrHmin)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)
    Q_h.set_by_ratio(Qmin, t_17, QrHmin)

    # Net Power
    if heating_cop_47 is None:
        heating_cop_47 = cop_47_h1_full(StagingType.SINGLE_STAGE, hspf2, Qm17rated)

    Pm17rated = Qm17rated * EIRm17rated
    PrHmin = QrHmin * EIRrHmin

    # 47 F
    P_h.set(Qrated, t_47, Q_h.get(Qrated, t_47) / heating_cop_47)
    P_h.set_by_ratio(Qmin, t_47, PrHmin)

    # 17 F
    P_h.set_by_maintenance(Qrated, t_17, t_47, Pm17rated)
    P_h.set_by_ratio(Qmin, t_17, PrHmin)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)


def make_packaged_terminal_model_data(
    cooling_capacity_95: float,  # Net total cooling capacity at 95F and rated speed
    eer_95: float,
    heating_capacity_47: float,
    heating_cop_47: float,
) -> TemperatureSpeedPerformance:
    Qrated = 1

    # COOLING

    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")

    t_c = [
        t_82,
        t_95,
    ]

    Qm95rated = 1.0 / calc_biquad(NRELDXModel.COOLING_CAP_FT_COEFFICIENTS, 67.0, 82.0)
    EIRm95rated = 1.0 / calc_biquad(NRELDXModel.COOLING_EIR_FT_COEFFICIENTS, 67.0, 82.0)

    Q_c = TemperatureSpeedCoolingPerformanceTable(t_c, 1)
    P_c = TemperatureSpeedCoolingPerformanceTable(t_c, 1)

    # Net Total Capacity

    # 95 F
    Q_c.set(Qrated, t_95, cooling_capacity_95)

    # 82 F
    Q_c.set_by_maintenance(Qrated, t_82, t_95, Qm95rated)

    # Net Power

    # 95/82 F
    P_c.set(Qrated, t_95, Q_c.get(Qrated, t_95) / fr_u(eer_95, "Btu/Wh"))

    Pm95rated = Qm95rated * EIRm95rated

    P_c.set_by_maintenance(Qrated, t_82, t_95, Pm95rated)

    # Tmin
    # Find temperature where power is half of 82F value to avoide division by zero

    t_c_min = P_c.get_temperature_at_value(P_c.get(1, t_82) * 0.5, t_82, t_95, 1)

    Q_c.add_temperature(t_c_min)
    P_c.add_temperature(t_c_min)

    t_40 = fr_u(40.0, "degF")
    if t_c_min > t_40:
        Q_c.add_temperature(t_40, extrapolate=True)
        P_c.add_temperature(t_40, extrapolate=False)

    # HEATING

    t_17 = fr_u(17, "degF")
    t_47 = fr_u(47, "degF")

    t_h = [
        t_17,
        t_47,
    ]

    EIRm17rated = calc_biquad(NRELDXModel.HEATING_EIR_FT_COEFFICIENTS, 70.0, 17.0)

    Qm17rated = calc_biquad(NRELDXModel.HEATING_CAP_FT_COEFFICIENTS, 70.0, 17.0)

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)

    # Net Power

    Pm17rated = Qm17rated * EIRm17rated

    # 47 F
    P_h.set(Qrated, t_47, Q_h.get(Qrated, t_47) / heating_cop_47)

    # 17 F
    P_h.set_by_maintenance(Qrated, t_17, t_47, Pm17rated)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()
    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)

from typing import Union, List
from bisect import insort
from copy import deepcopy

from scipy.interpolate import RegularGridInterpolator

from koozie import fr_u

from ..util import bracket, calc_biquad
from ..enums import StagingType

from .nrel import NRELDXModel

from .rating_correlations import cop_47_h1_full, cop_82_b_low


class TemperatureSpeedPerformanceTable:
    def __init__(
        self,
        temperatures: List[float],
        number_of_speeds: int,
        data_in: Union[List[List[Union[float, None]]], None] = None,
    ) -> None:
        self.temperatures: List[float] = deepcopy(temperatures)
        self.speeds: List[int] = [i + 1 for i in range(number_of_speeds)]
        self.data: List[List[Union[float, None]]]
        if data_in is None:
            self.data = [[None] * number_of_speeds for _ in range(len(temperatures))]
        else:
            self.data = data_in

        self.interpolator: RegularGridInterpolator

    def set(self, speed: int, temperature: float, value: float) -> None:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        self.data[temperature_index][speed_index] = value

    def get(self, speed: int, temperature: float) -> Union[float, None]:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        return self.data[temperature_index][speed_index]

    def set_by_ratio(self, speed: int, temperature: float, ratio: float) -> None:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        max_index = len(self.speeds) - 1
        if speed_index == max_index:
            # Set max from rated
            reference_speed_index = speed_index - 1
            self.data[temperature_index][speed_index] = (
                self.data[temperature_index][reference_speed_index] / ratio
            )
        else:
            # Set from max
            self.data[temperature_index][speed_index] = (
                self.data[temperature_index][max_index] * ratio
            )

    def get_ratio(self, speed: int, temperature: float) -> float:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        max_index = len(self.speeds) - 1
        return (
            self.data[temperature_index][speed_index]
            / self.data[temperature_index][max_index]
        )

    def set_by_interpolation(self, speed: int, temperature: float) -> None:
        """Interpolate rated speed from temperature above (currently only needed for lower temperatures)"""
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        s1 = speed_index
        s0 = s1 - 1
        s2 = s1 + 1
        t1 = temperature_index
        t2 = t1 + 1
        self.data[t1][s1] = self.data[t1][s0] + (
            self.data[t2][s1] - self.data[t2][s0]
        ) / (self.data[t2][s2] - self.data[t2][s0]) * (
            self.data[t1][s2] - self.data[t1][s0]
        )

    def add_temperature(
        self,
        temperature: float,
        extrapolate: bool = True,
        extrapolation_limit: Union[float, None] = None,
    ) -> None:
        if temperature in self.temperatures:
            raise RuntimeError(
                f"Temperature, {temperature:.2f}, already exists. Unable to add temperature."
            )
        insort(self.temperatures, temperature)
        t_i = self.temperatures.index(temperature)  # temperature index

        self.data.insert(t_i, [None] * len(self.speeds))

        if t_i == 0:
            # Extrapolate below
            t_0 = temperature
            t_1 = self.temperatures[1]
            t_2 = self.temperatures[2]
            for speed in self.speeds:
                s_i = speed - 1  # speed index
                if extrapolate:
                    value = self.data[t_i + 1][s_i] - (
                        self.data[t_i + 2][s_i] - self.data[t_i + 1][s_i]
                    ) / (t_2 - t_1) * (t_1 - t_0)
                    if extrapolation_limit is not None:
                        value = max(
                            value, self.data[t_i + 1][s_i] * extrapolation_limit
                        )
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
                    value = self.data[t_i - 1][s_i] + (
                        self.data[t_i - 1][s_i] - self.data[t_i - 2][s_i]
                    ) / (t_m1 - t_m2) * (t - t_m1)
                    if extrapolation_limit is not None:
                        value = min(
                            value, self.data[t_i - 1][s_i] * extrapolation_limit
                        )
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
                self.data[t_i][s_i] = self.data[t_i - 1][s_i] + (
                    self.data[t_i + 1][s_i] - self.data[t_i - 1][s_i]
                ) / (t_1 - t_m1) * (t - t_m1)

    def set_by_maintenance(
        self, speed: int, temperature: float, reference_temperature: float, ratio: float
    ) -> None:
        raise NotImplementedError()

    def get_maintenance(
        self, speed: int, temperature: float, reference_temperature: float
    ) -> float:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        reference_temperature_index = self.temperatures.index(reference_temperature)
        return (
            self.data[temperature_index][speed_index]
            / self.data[reference_temperature_index][speed_index]
        )

    def set_interpolator(self):
        self.interpolator = RegularGridInterpolator(
            (self.temperatures, self.speeds),
            self.data,
            "linear",
            False,
            None,
        )

    def calculate(self, speed: float, temperature: float) -> float:
        return self.interpolator([temperature, speed])[0]

    def apply_fan_power_correction(self, fan_powers: List[float]) -> None:
        assert len(fan_powers) == len(self.speeds)
        for t_i, speed_data in enumerate(self.data):
            for s_i, value in enumerate(speed_data):
                self.data[t_i][s_i] = value - fan_powers[s_i]
                assert self.data[t_i][s_i] > 0

        self.set_interpolator()


class TemperatureSpeedCoolingPerformanceTable(TemperatureSpeedPerformanceTable):
    def set_by_maintenance(
        self, speed: int, temperature: float, reference_temperature: float, ratio: float
    ) -> None:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        reference_temperature_index = self.temperatures.index(reference_temperature)
        if reference_temperature_index < temperature_index:
            self.data[temperature_index][speed_index] = (
                self.data[reference_temperature_index][speed_index] * ratio
            )
        else:
            self.data[temperature_index][speed_index] = (
                self.data[reference_temperature_index][speed_index] / ratio
            )


class TemperatureSpeedHeatingPerformanceTable(TemperatureSpeedPerformanceTable):
    def set_by_maintenance(
        self, speed: int, temperature: float, reference_temperature: float, ratio: float
    ) -> None:
        speed_index = speed - 1
        temperature_index = self.temperatures.index(temperature)
        reference_temperature_index = self.temperatures.index(reference_temperature)
        if reference_temperature_index > temperature_index:
            self.data[temperature_index][speed_index] = (
                self.data[reference_temperature_index][speed_index] * ratio
            )
        else:
            self.data[temperature_index][speed_index] = (
                self.data[reference_temperature_index][speed_index] / ratio
            )


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

    def cooling_capacity(self, speed: float = 2, temperature: float = fr_u(95, "degF")):
        return self.cooling_capacities.calculate(speed, temperature)

    def cooling_power(self, speed: float = 2, temperature: float = fr_u(95, "degF")):
        return self.cooling_powers.calculate(speed, temperature)

    def cooling_cop(self, speed: float = 2, temperature: float = fr_u(95, "degF")):
        return self.cooling_capacity(speed, temperature) / self.cooling_power(
            speed, temperature
        )

    def heating_capacity(self, speed: float = 2, temperature: float = fr_u(47, "degF")):
        return self.heating_capacities.calculate(speed, temperature)

    def heating_power(self, speed: float = 2, temperature: float = fr_u(47, "degF")):
        return self.heating_powers.calculate(speed, temperature)

    def heating_cop(self, speed: float = 2, temperature: float = fr_u(47, "degF")):
        return self.heating_capacity(speed, temperature) / self.heating_power(
            speed, temperature
        )

    def make_gross(
        self, cooling_fan_powers: List[float], heating_fan_powers: List[float]
    ) -> None:
        self.cooling_capacities.apply_fan_power_correction(
            [-p for p in cooling_fan_powers]
        )  # increases
        self.cooling_powers.apply_fan_power_correction(cooling_fan_powers)  # decreases
        self.heating_capacities.apply_fan_power_correction(
            heating_fan_powers
        )  # decreases
        self.heating_powers.apply_fan_power_correction(heating_fan_powers)  # decreases


def make_neep_statistical_model_data(
    cooling_capacity_95: float,  # Net total cooling capacity at 95F and rated speed
    seer2: float,
    eer2: float,
    heating_capacity_47: float,
    heating_capacity_17: float,
    hspf2: float,
    max_cooling_temperature: float = fr_u(125, "degF"),
    min_heating_temperature: float = fr_u(-20, "degF"),
    cooling_capacity_ratio: Union[float, None] = None,  # min/max capacity ratio at 95F
    cooling_cop_82_min: Union[float, None] = None,
    heating_cop_47: Union[float, None] = None,
) -> TemperatureSpeedPerformance:

    Qmin = 1
    Qrated = 2
    Qmax = 3

    t_60 = fr_u(60.0, "degF")

    # COOLING

    t_82 = fr_u(82.0, "degF")
    t_95 = fr_u(95.0, "degF")

    t_c = [
        t_82,
        t_95,
    ]

    Qr95rated = 0.934
    Qm95max = 0.940
    Qm95min = 0.948
    EIRr95rated = 0.928
    EIRm95max = 1.326
    EIRm95min = 1.315

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
        cooling_cop_82_min = cop_82_b_low(
            StagingType.VARIABLE_SPEED, seer2, seer2 / eer2
        )
    else:
        EIRr82min = (Q_c.get(Qmax, t_82) / P_c.get(Qmax, t_82)) / cooling_cop_82_min

    EIRr82min = (Q_c.get(Qmax, t_82) / P_c.get(Qmax, t_82)) / cooling_cop_82_min

    if cooling_capacity_ratio is not None:
        Qr95min = cooling_capacity_ratio
    else:
        Qr95min = bracket(
            0.510 - 0.119 * (EIRr82min - 1.305) / (-0.324), 0.1, Qr95rated
        )

    # Back to capacities
    Q_c.set_by_ratio(Qmin, t_95, Qr95min)
    Q_c.set_by_maintenance(Qmin, t_82, t_95, Qm95min)
    Q_c.set_by_interpolation(Qrated, t_82)

    P_c.set(Qmin, t_82, Q_c.get(Qmin, t_82) / cooling_cop_82_min)

    P_c.set_by_maintenance(Qmin, t_95, t_82, Pm95min)
    P_c.set_by_interpolation(Qrated, t_82)

    # Tmin
    Q_c.add_temperature(t_60, extrapolation_limit=0.5)
    P_c.add_temperature(t_60, extrapolation_limit=0.5)

    # Tmax
    Q_c.add_temperature(max_cooling_temperature)
    P_c.add_temperature(max_cooling_temperature)

    # HEATING

    t_min = min_heating_temperature
    t_5 = fr_u(5, "degF")
    t_17 = fr_u(17, "degF")
    t_47 = fr_u(47, "degF")

    t_h = [
        t_min,
        t_5,
        t_17,
        t_47,
    ]

    Qr47rated = 0.908
    Qr47min = 0.272
    Qr17rated = 0.817
    Qr17min = 0.341
    Qm5max = 0.866
    Qr5rated = 0.988
    Qr5min = 0.321
    QmslopeLCTmax = -0.025
    QmslopeLCTmin = -0.024
    EIRr47rated = 0.939
    EIRr47min = 0.730
    EIRm17rated = 1.351
    EIRr17rated = 0.902
    EIRr17min = 0.798
    EIRm5max = 1.164
    EIRr5rated = 1.000
    EIRr5min = 0.866
    EIRmslopeLCTmax = 0.012
    EIRmslopeLCTmin = 0.012

    if heating_capacity_17 is not None:
        Qm17rated = heating_capacity_17 / heating_capacity_47
    else:
        Qm17rated = 0.689

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

    QmLCTmax = 1 / (1 - fr_u(QmslopeLCTmax, "1/degF") * (t_5 - t_min))
    QmLCTmin = 1 / (1 - fr_u(QmslopeLCTmin, "1/degF") * (t_5 - t_min))

    # Tmin
    Q_h.set_by_maintenance(Qmax, t_min, t_5, QmLCTmax)
    Q_h.set_by_maintenance(Qmin, t_min, t_5, QmLCTmin)
    Q_h.set_by_interpolation(Qrated, t_min)

    # Tmax
    Q_h.add_temperature(t_60, False)

    # Net Power
    if heating_cop_47 is None:
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

    EIRmLCTmax = 1 / (1 - fr_u(EIRmslopeLCTmax, "1/degF") * (t_5 - t_min))
    EIRmLCTmin = 1 / (1 - fr_u(EIRmslopeLCTmin, "1/degF") * (t_5 - t_min))

    PmLCTmax = EIRmLCTmax * Q_h.get_maintenance(Qmax, t_min, t_5)
    PmLCTmin = EIRmLCTmin * Q_h.get_maintenance(Qmin, t_min, t_5)

    # Tmin
    P_h.set_by_maintenance(Qmax, t_min, t_5, PmLCTmax)
    P_h.set_by_maintenance(Qmin, t_min, t_5, PmLCTmin)
    P_h.set_by_interpolation(Qrated, t_min)

    # Tmax
    P_h.add_temperature(t_60, False)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)


def make_neep_model_data(
    cooling_capacities: List[List[Union[float, None]]],
    cooling_powers: List[List[Union[float, None]]],
    heating_capacities: List[List[Union[float, None]]],
    heating_powers: List[List[Union[float, None]]],
    lct: Union[float, None] = None,
    max_cooling_temperature: float = fr_u(125, "degF"),
    min_heating_temperature: float = fr_u(0, "degF"),
):
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
    t_60 = fr_u(60.0, "degF")

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

    Q_c = TemperatureSpeedCoolingPerformanceTable(
        cooling_temperatures, 3, cooling_capacities
    )
    P_c = TemperatureSpeedCoolingPerformanceTable(
        cooling_temperatures, 3, cooling_powers
    )
    Q_h = TemperatureSpeedHeatingPerformanceTable(
        heating_temperatures, 3, heating_capacities
    )
    P_h = TemperatureSpeedHeatingPerformanceTable(
        heating_temperatures, 3, heating_powers
    )

    # Interpolate for missing rated conditions, and extrapolate to extreme temperatures
    Q_c.set_by_interpolation(2, t_82)
    P_c.set_by_interpolation(2, t_82)

    Q_c.add_temperature(t_60, extrapolation_limit=0.5)
    P_c.add_temperature(t_60, extrapolation_limit=0.5)

    Q_c.add_temperature(max_cooling_temperature)
    P_c.add_temperature(max_cooling_temperature)

    if Q_h.get(2, t_5) is None:
        Q_h.set_by_interpolation(2, t_5)

    if P_h.get(2, t_5) is None:
        P_h.set_by_interpolation(2, t_5)

    if lct is not None:
        Q_h.set_by_interpolation(2, t_lct)
        P_h.set_by_interpolation(2, t_lct)
        if min_heating_temperature < t_lct:
            Q_h.add_temperature(min_heating_temperature)
            P_h.add_temperature(min_heating_temperature)
    else:
        Q_h.add_temperature(min_heating_temperature)
        P_h.add_temperature(min_heating_temperature)

    Q_h.add_temperature(t_60, False)
    P_h.add_temperature(t_60, False)

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
    heating_capacity_17: Union[float, None],
    hspf2: float,
    max_cooling_temperature: float = fr_u(125, "degF"),
    min_heating_temperature: float = fr_u(-20, "degF"),
    heating_cop_47: Union[float, None] = None,
    cycling_degradation_coefficient: float = 0.15,
) -> TemperatureSpeedPerformance:
    Qrated = 1

    t_60 = fr_u(60.0, "degF")

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
    eer2_b = seer2 / (
        1.0 - 0.5 * cycling_degradation_coefficient
    )  # EER2 at B (82F) conditions
    P_c.set(Qrated, t_82, Q_c.get(Qrated, t_82) / fr_u(eer2_b, "Btu/Wh"))

    # Tmin
    Q_c.add_temperature(t_60, extrapolation_limit=0.5)
    P_c.add_temperature(t_60, extrapolation_limit=0.5)

    # Tmax
    Q_c.add_temperature(max_cooling_temperature)
    P_c.add_temperature(max_cooling_temperature)

    # HEATING

    t_min = min_heating_temperature
    t_5 = fr_u(5, "degF")
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
        Qm17rated = (
            0.626  # Based on AHRI directory units believed to be single speed (4/4/24)
        )

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 1)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)

    # 5 F
    Q_h.add_temperature(t_5, True)

    # Tmin
    Q_h.add_temperature(t_min, True)

    # Tmax
    Q_h.add_temperature(t_60, False)

    # Net Power
    if heating_cop_47 is None:
        heating_cop_47 = cop_47_h1_full(StagingType.SINGLE_STAGE, hspf2, Qm17rated)

    Pm17rated = Qm17rated * EIRm17rated

    # 47 F
    P_h.set(Qrated, t_47, Q_h.get(Qrated, t_47) / heating_cop_47)

    # 17 F
    P_h.set_by_maintenance(Qrated, t_17, t_47, Pm17rated)

    # 5 F
    P_h.add_temperature(t_5)

    # Tmin
    P_h.add_temperature(t_min)

    # Tmax
    P_h.add_temperature(t_60, extrapolate=False)

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
    heating_capacity_17: Union[float, None],
    hspf2: float,
    max_cooling_temperature: float = fr_u(125, "degF"),
    min_heating_temperature: float = fr_u(-20, "degF"),
    cooling_cop_82_min: Union[float, None] = None,
    heating_cop_47: Union[float, None] = None,
) -> TemperatureSpeedPerformance:
    Qmin = 1
    Qrated = 2

    t_60 = fr_u(60.0, "degF")

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
        cooling_cop_82_min = cop_82_b_low(
            StagingType.VARIABLE_SPEED, seer2, seer2 / eer2
        )

    # 82 / 95 F
    P_c.set(Qrated, t_95, Q_c.get(Qrated, t_95) / fr_u(eer2, "Btu/Wh"))
    P_c.set_by_maintenance(Qrated, t_82, t_95, Pm95rated)
    P_c.set(Qmin, t_82, Q_c.get(Qmin, t_82) / cooling_cop_82_min)
    P_c.set_by_maintenance(Qmin, t_95, t_82, Pm95rated)

    # Tmin
    Q_c.add_temperature(t_60, extrapolation_limit=0.5)
    P_c.add_temperature(t_60, extrapolation_limit=0.5)

    # Tmax
    Q_c.add_temperature(max_cooling_temperature)
    P_c.add_temperature(max_cooling_temperature)

    # HEATING

    t_min = min_heating_temperature
    t_5 = fr_u(5, "degF")
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
        Qm17rated = 0.626  # Based on AHRI directory units believed to be single speed (4/4/24) TODO: Switch to Cutler curve

    Q_h = TemperatureSpeedHeatingPerformanceTable(t_h, 2)
    P_h = TemperatureSpeedHeatingPerformanceTable(t_h, 2)

    # Net Total Capacity

    # 47 F
    Q_h.set(Qrated, t_47, heating_capacity_47)
    Q_h.set_by_ratio(Qmin, t_47, QrHmin)

    # 17 F
    Q_h.set_by_maintenance(Qrated, t_17, t_47, Qm17rated)
    Q_h.set_by_ratio(Qmin, t_17, QrHmin)

    # 5 F
    Q_h.add_temperature(t_5, True)

    # Tmin
    Q_h.add_temperature(t_min, True)

    # Tmax
    Q_h.add_temperature(t_60, False)

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

    # 5 F
    P_h.add_temperature(t_5)

    # Tmin
    P_h.add_temperature(t_min)

    # Tmax
    P_h.add_temperature(t_60, extrapolate=False)

    Q_c.set_interpolator()
    P_c.set_interpolator()
    Q_h.set_interpolator()
    P_h.set_interpolator()

    return TemperatureSpeedPerformance(Q_c, P_c, Q_h, P_h)

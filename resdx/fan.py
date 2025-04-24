import uuid
import datetime
from random import Random
from typing import List, Tuple
from enum import Enum

from math import exp, log, inf
from scipy import optimize  # Used for finding system/fan curve intersection
from numpy import linspace, array
from koozie import fr_u, to_u, convert

from .util import calc_quad


class FanMetadata:
    def __init__(
        self,
        description="",
        data_source="https://github.com/bigladder/resdx",
        notes="",
        uuid_seed=None,
        data_version=1,
    ):
        self.description = description
        self.data_source = data_source
        self.notes = notes
        self.uuid_seed = uuid_seed
        self.data_version = data_version


class FanMotorType(Enum):
    UNKNOWN = 0
    PSC = 1
    BPM = 2


class Fan:
    """Base class for fan models"""

    SYSTEM_EXPONENT = 0.5

    def __init__(
        self,
        design_airflow,
        design_external_static_pressure,
        design_efficacy=fr_u(0.365, "W/cfm"),  # AHRI 210/240 2017 default
        metadata=None,
    ):
        self.design_external_static_pressure = design_external_static_pressure
        self.design_efficacy = design_efficacy
        self.number_of_speeds = 0
        self.design_airflow = []
        self.design_airflow_ratio = []
        self.fan_motor_type = FanMotorType.UNKNOWN

        if metadata is None:
            self.metadata = FanMetadata()
        else:
            self.metadata = metadata

        if type(design_airflow) is list:
            self.calculate_system_curve_constant(design_airflow[0])
            for airflow in design_airflow:
                self.add_speed(airflow)
        else:
            self.calculate_system_curve_constant(design_airflow)
            self.add_speed(design_airflow)

    def add_speed(self, airflow, external_static_pressure=None):
        self.design_airflow.append(airflow)
        self.number_of_speeds += 1
        self.design_airflow_ratio.append(self.design_airflow[-1] / self.design_airflow[0])

    def remove_speed(self, speed_setting):
        self.design_airflow.pop(speed_setting)
        self.number_of_speeds -= 1
        self.design_airflow_ratio.pop(speed_setting)

    def system_pressure(self, airflow):
        return (airflow * self.system_curve_constant) ** (1.0 / self.SYSTEM_EXPONENT)

    def system_flow(self, external_static_pressure):
        return external_static_pressure ** (self.SYSTEM_EXPONENT) / self.system_curve_constant

    def calculate_system_curve_constant(self, nominal_airflow: float) -> None:
        self.system_curve_constant = self.design_external_static_pressure ** (self.SYSTEM_EXPONENT) / nominal_airflow

    def efficacy(self, speed_setting, external_static_pressure=None):
        raise NotImplementedError()

    def airflow(self, speed_setting, external_static_pressure=None):
        raise NotImplementedError()

    def airflow_ratio(self, speed_setting, base_speed=0, external_static_pressure=None):
        return self.airflow(speed_setting, external_static_pressure) / self.airflow(
            base_speed, external_static_pressure
        )

    def power(self, speed_setting, external_static_pressure=None):
        return self.airflow(speed_setting, external_static_pressure) * self.efficacy(
            speed_setting, external_static_pressure
        )

    def power_ratio(self, speed_setting, base_speed=0, external_static_pressure=None):
        return self.power(speed_setting, external_static_pressure) / self.power(base_speed, external_static_pressure)

    def fluid_power(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            return self.airflow(speed_setting, external_static_pressure) * self.operating_pressure(speed_setting)
        else:
            return self.airflow(speed_setting, external_static_pressure) * external_static_pressure

    def efficiency(self, speed_setting, external_static_pressure=None):
        return self.fluid_power(speed_setting, external_static_pressure) / self.power(
            speed_setting, external_static_pressure
        )

    def rotational_speed(self, speed_setting, external_static_pressure=None):
        # Reasonable default. Not needed for much beyond 205 representations.
        return fr_u(1100, "rpm") * self.airflow_ratio(speed_setting, 0, external_static_pressure)

    def operating_pressure(self, speed_setting, system_curve=None):
        # Calculate pressure that corresponds to intersection of system curve and fan curve for this setting
        if system_curve is None:
            fx = self.system_flow
        else:
            fx = system_curve
        p, solution = optimize.brentq(
            lambda x: self.airflow(speed_setting, x) - fx(x),
            0.0,
            3.0 * self.design_external_static_pressure,
            full_output=True,
        )
        return p

    def check_power(self, airflow, external_static_pressure=None):
        self.add_speed(airflow)
        new_speed_setting = self.number_of_speeds - 1
        power = self.power(new_speed_setting, external_static_pressure)
        self.remove_speed(new_speed_setting)
        return power

    def find_rated_fan_speed(
        self,
        gross_capacity,
        rated_flow_per_rated_net_capacity,
        guess_airflow=None,
        rated_full_flow_external_static_pressure=None,
        cooling=True,
    ):
        """Given a gross capacity, and the rated flow per rated net capacity, find the speed and flow rate that gives consistent results"""
        """Q_gross +/- Q_fan = Q_net"""
        if guess_airflow is None:
            guess_airflow = gross_capacity * rated_flow_per_rated_net_capacity

        if rated_full_flow_external_static_pressure is not None:
            full_airflow = self.airflow(0, rated_full_flow_external_static_pressure)

            def pressure_function(airflow):
                return rated_full_flow_external_static_pressure * (airflow / full_airflow) ** 2

        else:

            def pressure_function(airflow):
                return None

        if cooling:

            def net_capacity_function(airflow):
                return gross_capacity - self.check_power(airflow, pressure_function(airflow))

        else:

            def net_capacity_function(airflow):
                return gross_capacity + self.check_power(airflow, pressure_function(airflow))

        def optimization_function(airflow):
            return net_capacity_function(airflow) - airflow / rated_flow_per_rated_net_capacity

        airflow, solution = optimize.newton(optimization_function, guess_airflow, full_output=True)
        self.add_speed(airflow, pressure_function(airflow))
        return airflow

    def get_speed_order_map(self):
        airflow_list = array([self.airflow(speed, 0.0) for speed in range(self.number_of_speeds)])
        return airflow_list.argsort()

    def generate_205_representation(self):
        timestamp = datetime.datetime.now().isoformat("T", "minutes")
        rnd = Random()
        if self.metadata.uuid_seed is None:
            self.metadata.uuid_seed = hash(self)
        rnd.seed(self.metadata.uuid_seed)
        unique_id = str(uuid.UUID(int=rnd.getrandbits(128), version=4))

        speed_order_map = self.get_speed_order_map()
        max_speed = speed_order_map[-1]

        if type(self) is ECMFlowFan:
            if self.maximum_power is inf:
                # Assumption
                max_power = (
                    self.power(
                        speed_setting=max_speed,
                        external_static_pressure=fr_u(1.2, "in_H2O"),
                    )
                    * 1.2
                )
            else:
                max_power = self.maximum_power
        else:
            # Assumption
            max_power = self.power(speed_setting=max_speed, external_static_pressure=0.0) * 1.2

        # RS0005 Motor
        rnd.seed(unique_id)

        metadata_motor = {
            "data_model": "ASHRAE_205",
            "schema": "RS0005",
            "schema_version": "2.0.0",
            "description": f"Placeholder motor representation (performance characterized in parent RS0003 fan assembly)",
            "id": str(uuid.UUID(int=rnd.getrandbits(128), version=4)),
            "data_timestamp": f"{timestamp}Z",
            "data_version": self.metadata.data_version,
            "data_source": self.metadata.data_source,
            "disclaimer": "This data is synthetic and does not represent any physical products.",
        }

        performance_motor = {
            "maximum_power": max_power,
            "standby_power": 0.0,
            "number_of_poles": 6,
        }

        design_airflow = self.design_airflow[0] if type(self.design_airflow) is list else self.design_airflow
        design_efficacy = self.design_efficacy[0] if type(self.design_efficacy) is list else self.design_efficacy
        design_external_static_pressure = (
            self.design_external_static_pressure[0]
            if type(self.design_external_static_pressure) is list
            else self.design_external_static_pressure
        )

        if len(self.metadata.description) == 0:
            airflow_cfm = to_u(design_airflow, "cfm")
            efficacy_w_cfm = to_u(design_efficacy, "W/cfm")
            pressure_in_h2o = to_u(design_external_static_pressure, "in_H2O")
            if type(self) is PSCFan:
                fan_description = "Permanent Split Capacitor (PSC) fan"
            elif type(self) is ECMFlowFan:
                fan_description = f"Electronically Commutated Motor (ECM) fan"
            else:
                fan_description = "fan"
            self.metadata.description = (
                f"{airflow_cfm:.0f} cfm {fan_description} ({efficacy_w_cfm:.3f} W/cfm @ {pressure_in_h2o:.2f} in. H2O)"
            )

        metadata = {
            "data_model": "ASHRAE_205",
            "schema": "RS0003",
            "schema_version": "2.0.0",
            "description": self.metadata.description,
            "id": unique_id,
            "data_timestamp": f"{timestamp}Z",
            "data_version": self.metadata.data_version,
            "data_source": self.metadata.data_source,
            "disclaimer": "This data is synthetic and does not represent any physical products.",
        }

        if len(self.metadata.notes) > 0:
            metadata["notes"] = self.metadata.notes

        # Create conditions
        speed_number = list(range(1, self.number_of_speeds + 1))

        if type(self) is PSCFan:
            max_static_pressure = self.block_pressure[max_speed]
        else:
            max_static_pressure = fr_u(1.2, "in_H2O")
        static_pressure_difference = linspace(fr_u(0.0, "in_H2O"), max_static_pressure, 6).tolist()

        grid_variables = {
            "speed_number": speed_number,
            "static_pressure_difference": static_pressure_difference,
        }

        standard_air_volumetric_flow_rate = []
        shaft_power = []
        impeller_rotational_speed = []
        operation_state = []

        for speed in speed_number:
            for esp in static_pressure_difference:
                # Get speed number from ordered number
                positional_speed_number = speed_order_map[speed - 1]
                shaft_power.append(
                    self.power(
                        speed_setting=positional_speed_number,
                        external_static_pressure=esp,
                    )
                )
                standard_air_volumetric_flow_rate.append(
                    self.airflow(
                        speed_setting=positional_speed_number,
                        external_static_pressure=esp,
                    )
                )
                impeller_rotational_speed.append(
                    to_u(
                        self.rotational_speed(
                            speed_setting=positional_speed_number,
                            external_static_pressure=esp,
                        ),
                        "rps",
                    )
                )
                operation_state.append("NORMAL")

        performance_map = {
            "grid_variables": grid_variables,
            "lookup_variables": {
                "shaft_power": shaft_power,
                "standard_air_volumetric_flow_rate": standard_air_volumetric_flow_rate,
                "impeller_rotational_speed": impeller_rotational_speed,
                "operation_state": operation_state,
            },
        }

        performance = {
            "nominal_standard_air_volumetric_flow_rate": self.airflow(
                speed_setting=max_speed,
                external_static_pressure=self.design_external_static_pressure,
            ),
            "is_enclosed": True,
            "assembly_components": [
                {"component_type": "COIL", "wet_pressure_difference": 75.0}  # Pa
            ],
            "heat_loss_fraction": 1.0,
            "maximum_impeller_rotational_speed": convert(1500.0, "rpm", "rps"),
            "minimum_impeller_rotational_speed": 0.0,
            "operation_speed_control_type": "DISCRETE",
            "installation_speed_control_type": "FIXED",
            "motor_representation": {
                "metadata": metadata_motor,
                "performance": performance_motor,
            },
            "performance_map": performance_map,
        }

        representation = {"metadata": metadata, "performance": performance}

        return representation


class ConstantEfficacyFan(Fan):
    def __init__(
        self,
        design_airflow,
        design_external_static_pressure,
        design_efficacy=fr_u(0.365, "W/cfm"),
    ):
        super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
        if type(self.design_efficacy) is not list:
            self.design_efficacy = [self.design_efficacy] * self.number_of_speeds

    def add_speed(self, airflow, efficacy=None, external_static_pressure=None):
        super().add_speed(airflow, external_static_pressure)
        if efficacy is not None:
            self.design_efficacy.append(efficacy)

    def remove_speed(self, speed_setting):
        super().remove_speed(speed_setting)
        self.design_efficacy.pop(speed_setting)

    def efficacy(self, speed_setting, external_static_pressure=None):
        return self.design_efficacy[speed_setting]

    def airflow(self, speed_setting, external_static_pressure=None):
        return self.design_airflow[speed_setting]


class PSCFan(Fan):
    """Based largely on measured fan performance by Proctor Engineering"""

    """Model needs more data to refine and further generalize"""

    AIRFLOW_COEFFICIENT = fr_u(10.0, "cfm")
    AIRFLOW_EXP_COEFFICIENT = fr_u(5.35, "1/in_H2O")
    EFFICACY_SLOPE = 0.3  # Relative change in efficacy at lower flow ratios (data is fairly inconsistent on this value)

    def __init__(
        self,
        design_airflow,
        design_external_static_pressure=fr_u(0.5, "in_H2O"),
        design_efficacy=fr_u(0.365, "W/cfm"),
    ):
        self.design_airflow_reduction = self.airflow_reduction(design_external_static_pressure)
        self.free_airflow = []
        self.free_airflow_ratio = []
        self.speed_efficacy = []
        self.block_pressure = []
        self.free_speed = []
        super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
        self.fan_motor_type = FanMotorType.PSC

    def add_speed(self, airflow, external_static_pressure=None):
        if external_static_pressure is not None:
            design_airflow = airflow - (
                self.design_airflow_reduction - self.airflow_reduction(external_static_pressure)
            )
        else:
            design_airflow = airflow
        super().add_speed(design_airflow)
        self.free_airflow.append(self.design_airflow[-1] + self.design_airflow_reduction)
        self.free_airflow_ratio.append(self.free_airflow[-1] / self.free_airflow[0])
        self.speed_efficacy.append(
            self.design_efficacy * (1.0 + self.EFFICACY_SLOPE * (self.free_airflow_ratio[-1] - 1.0))
        )
        self.block_pressure.append(
            log(self.free_airflow[-1] / self.AIRFLOW_COEFFICIENT + 1.0) / self.AIRFLOW_EXP_COEFFICIENT
        )
        self.free_speed.append(fr_u(1040.0, "rpm") * self.free_airflow_ratio[-1])

    def remove_speed(self, speed_setting):
        super().remove_speed(speed_setting)
        self.free_airflow.pop(speed_setting)
        self.free_airflow_ratio.pop(speed_setting)
        self.speed_efficacy.pop(speed_setting)
        self.block_pressure.pop(speed_setting)
        self.free_speed.pop(speed_setting)

    def efficacy(self, speed_setting, external_static_pressure=None):
        return self.speed_efficacy[speed_setting]

    def airflow(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        return max(
            self.free_airflow[speed_setting] - self.airflow_reduction(external_static_pressure),
            0.0,
        )

    def rotational_speed(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        if external_static_pressure > self.block_pressure[speed_setting]:
            return fr_u(1100.0, "rpm")
        else:
            i = speed_setting
            return (
                self.free_speed[i]
                + (fr_u(1100.0, "rpm") - self.free_speed[i]) * external_static_pressure / self.block_pressure[i]
            )

    def airflow_reduction(self, external_static_pressure):
        return self.AIRFLOW_COEFFICIENT * (exp(external_static_pressure * self.AIRFLOW_EXP_COEFFICIENT) - 1.0)

    def operating_pressure(self, speed_setting, system_curve=None):
        if system_curve is None:
            # TODO Solve algebraically for improved performance
            return super().operating_pressure(speed_setting, system_curve)
        else:
            return super().operating_pressure(speed_setting, system_curve)

        # Solve algebraically
        pass


class ECMFlowFan(Fan):
    """Constant flow ECM fan. Based largely on measured fan performance by Proctor Engineering"""

    EFFICACY_SLOPE_ESP = fr_u(
        0.235, "(W/cfm)/in_H2O"
    )  # Relative change in efficacy at different external static pressures
    SPEED_SLOPE_ESP = fr_u(
        463.5, "rpm/in_H2O"
    )  # Relative change in rotational speed at different external static pressures

    def __init__(
        self,
        design_airflow,
        design_external_static_pressure=fr_u(0.5, "in_H2O"),
        design_efficacy=fr_u(0.365, "W/cfm"),
        maximum_power=inf,
    ):
        # Check if design power is above power limit
        design_power = (design_airflow[0] if type(design_airflow) is list else design_airflow) * design_efficacy
        if design_power > maximum_power:
            raise RuntimeError(f"Design power ({design_power} W) is greater than the maximum power ({maximum_power}) W")
        self.maximum_power = maximum_power
        self.design_free_efficacy = design_efficacy - self.EFFICACY_SLOPE_ESP * design_external_static_pressure
        self.free_efficacy = []
        super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
        self.fan_motor_type = FanMotorType.BPM

    def add_speed(self, airflow, external_static_pressure=None):
        super().add_speed(airflow, external_static_pressure)
        self.free_efficacy.append(
            self.design_free_efficacy * self.design_airflow_ratio[-1] * self.design_airflow_ratio[-1]
        )

    def remove_speed(self, speed_setting):
        super().remove_speed(speed_setting)
        self.free_efficacy.pop(speed_setting)

    def unconstrained_efficacy(self, speed_setting, external_static_pressure):
        return self.free_efficacy[speed_setting] + self.EFFICACY_SLOPE_ESP * external_static_pressure

    def unconstrained_power(self, speed_setting, external_static_pressure):
        return self.design_airflow[speed_setting] * self.unconstrained_efficacy(speed_setting, external_static_pressure)

    def power(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        return min(
            self.unconstrained_power(speed_setting, external_static_pressure),
            self.maximum_power,
        )

    def airflow(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        if external_static_pressure == 0.0:
            return self.design_airflow[speed_setting]
        else:
            estimated_flow_power = (
                self.design_airflow[speed_setting]
                * external_static_pressure
                * (
                    self.power(speed_setting, external_static_pressure)
                    / self.unconstrained_power(speed_setting, external_static_pressure)
                )
                ** 0.5
            )
            return estimated_flow_power / external_static_pressure

    def efficacy(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        return self.power(speed_setting, external_static_pressure) / self.airflow(
            speed_setting, external_static_pressure
        )

    def unconstrained_rotational_speed(self, speed_setting, external_static_pressure):
        return (
            fr_u(1100.0, "rpm")
            - self.SPEED_SLOPE_ESP * (self.design_external_static_pressure - external_static_pressure)
        ) * self.design_airflow_ratio[speed_setting]

    def rotational_speed(self, speed_setting, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.operating_pressure(speed_setting)
        return self.unconstrained_rotational_speed(speed_setting, external_static_pressure) * (
            self.efficacy(speed_setting, external_static_pressure)
            / self.unconstrained_efficacy(speed_setting, external_static_pressure)
        )


class EEREFan(Fan):
    """Base class for fans modeled with 'EERE-2014-BT-STD-0048-0098' assumptions"""

    NOMINAL_CAPACITIES = [fr_u(capacity, "ton_ref") for capacity in (2.0, 3.0, 4.0, 5.0)]
    FLOW_COEFFICIENTS: Tuple[float, float]
    BASE_EFFICACIES: List[float]
    EFFICACY_COEFFICIENTS: Tuple[float, float]

    def __init__(
        self,
        design_airflow,
        design_external_static_pressure,
    ):
        self.free_airflow: List[float] = []
        super().__init__(design_airflow, design_external_static_pressure)
        self.nominal_system_capacity = self.design_airflow[0] / fr_u(400.0, "cfm/ton_ref")

        # Determine indices of sizes to interpolate/extrapolate from
        if self.nominal_system_capacity < self.NOMINAL_CAPACITIES[0]:
            self.lower_index = 0
            self.upper_index = 1
        elif self.nominal_system_capacity > self.NOMINAL_CAPACITIES[len(self.NOMINAL_CAPACITIES) - 1]:
            self.lower_index = len(self.NOMINAL_CAPACITIES) - 2
            self.upper_index = len(self.NOMINAL_CAPACITIES) - 1
        else:
            for index, capacity in enumerate(self.NOMINAL_CAPACITIES):
                if self.nominal_system_capacity > capacity:
                    self.lower_index = index
                    self.upper_index = index + 1

        self.capacity_weight = (self.nominal_system_capacity - self.NOMINAL_CAPACITIES[self.lower_index]) / (
            self.NOMINAL_CAPACITIES[self.upper_index] - self.NOMINAL_CAPACITIES[self.lower_index]
        )

    def airflow(self, speed_setting, external_static_pressure=None):
        return calc_quad(
            (self.free_airflow[speed_setting], *self.FLOW_COEFFICIENTS),
            external_static_pressure,
        )

    def efficacy(self, speed_setting, external_static_pressure=None):
        """Calculate efficacy based on Eq. 7-C.4 in 'EERE-2014-BT-STD-0048-0098' using linear interpolation"""
        lower_coefficients = (
            self.BASE_EFFICACIES[self.lower_index],
            *self.EFFICACY_COEFFICIENTS,
        )
        upper_coefficients = (
            self.BASE_EFFICACIES[self.upper_index],
            *self.EFFICACY_COEFFICIENTS,
        )
        efficacy_lower = calc_quad(lower_coefficients, external_static_pressure)
        efficacy_upper = calc_quad(upper_coefficients, external_static_pressure)

        return efficacy_lower + (efficacy_upper - efficacy_lower) * self.capacity_weight

    def add_speed(self, airflow, external_static_pressure=None):
        if external_static_pressure is None:
            external_static_pressure = self.system_pressure(airflow)
        super().add_speed(airflow, external_static_pressure)
        airflow_offset = (
            self.FLOW_COEFFICIENTS[0] * external_static_pressure
            + self.FLOW_COEFFICIENTS[1] * external_static_pressure * external_static_pressure
        )
        self.free_airflow.append(airflow - airflow_offset)

    def remove_speed(self, speed_setting):
        super().remove_speed(speed_setting)
        self.free_airflow.pop(speed_setting)

    def rotational_speed(self, speed_setting, external_static_pressure=None):
        return super().rotational_speed(speed_setting, external_static_pressure)


class EEREBaselinePSCFan(EEREFan):
    FLOW_COEFFICIENTS = (fr_u(49.0, "cfm/in_H2O"), fr_u(-570.0, "cfm/in_H2O**2"))
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.49, 0.52, 0.55, 0.57)]
    EFFICACY_COEFFICIENTS = (
        fr_u(-0.2, "(W/cfm)/in_H2O"),
        fr_u(0.19, "(W/cfm)/in_H2O**2"),
    )

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.PSC


class EEREImprovedPSCFan(EEREBaselinePSCFan):
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.44, 0.47, 0.49, 0.52)]

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.PSC


class EEREPSCWithControlsFan(EEREFan):
    FLOW_COEFFICIENTS = (fr_u(267.0, "cfm/in_H2O"), fr_u(-338.0, "cfm/in_H2O**2"))
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.25, 0.27, 0.29, 0.31)]
    EFFICACY_COEFFICIENTS = (
        fr_u(0.14, "(W/cfm)/in_H2O"),
        fr_u(0.06, "(W/cfm)/in_H2O**2"),
    )

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.PSC


class EEREBPMSingleStageConstantTorqueFan(EEREFan):
    FLOW_COEFFICIENTS = (fr_u(-456.0, "cfm/in_H2O"), fr_u(8.0, "cfm/in_H2O**2"))
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.18, 0.19, 0.21, 0.23)]
    EFFICACY_COEFFICIENTS = (
        fr_u(0.12, "(W/cfm)/in_H2O"),
        fr_u(0.07, "(W/cfm)/in_H2O**2"),
    )

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.BPM


class EEREBPMMultiStageConstantTorqueFan(EEREBPMSingleStageConstantTorqueFan):
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.14, 0.15, 0.17, 0.16)]

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.BPM


class EEREBPMMultiStageConstantAirflowFan(EEREFan):
    FLOW_COEFFICIENTS = (fr_u(99.0, "cfm/in_H2O"), fr_u(-103.0, "cfm/in_H2O**2"))
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.11, 0.12, 0.13, 0.15)]
    EFFICACY_COEFFICIENTS = (
        fr_u(0.25, "(W/cfm)/in_H2O"),
        fr_u(-0.01, "(W/cfm)/in_H2O**2"),
    )

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.BPM


class EEREBPMMultiStageBackwardCurvedImpellerConstantAirflowFan(EEREBPMMultiStageConstantAirflowFan):
    BASE_EFFICACIES = [fr_u(v, "W/cfm") for v in (0.09, 0.10, 0.11, 0.12)]

    def __init__(self, design_airflow, design_external_static_pressure):
        super().__init__(design_airflow, design_external_static_pressure)
        self.fan_motor_type = FanMotorType.BPM


# RESNET Fan Models
class RESNETFan(Fan):
    def airflow(self, speed_setting, external_static_pressure=None):
        return self.design_airflow[speed_setting]


class RESNETPSCFan(RESNETFan):
    EFFICACY_SLOPE = PSCFan.EFFICACY_SLOPE

    def __init__(
        self,
        design_airflow,
    ):
        super().__init__(design_airflow, fr_u(0.5, "in_H2O"), fr_u(0.414, "W/cfm"))
        self.fan_motor_type = FanMotorType.PSC

    def efficacy(self, speed_setting, external_static_pressure=None):
        return self.design_efficacy * (
            self.EFFICACY_SLOPE * self.airflow_ratio(speed_setting, 0, external_static_pressure)
            + (1.0 - self.EFFICACY_SLOPE)
        )


class RESNETBPMFan(RESNETFan):
    DUCTED_DESIGN_EFFICACY = fr_u(0.281, "W/cfm")
    DUCTLESS_DESIGN_EFFICACY = fr_u(0.171, "W/cfm")

    def __init__(
        self,
        design_airflow,
    ):
        super().__init__(design_airflow, fr_u(0.5, "in_H2O"), self.DUCTED_DESIGN_EFFICACY)
        self.fan_motor_type = FanMotorType.BPM

    def efficacy(self, speed_setting, external_static_pressure=None):
        ducted_external_static_pressure = self.operating_pressure(speed_setting)
        ductless_external_static_pressure = 0.0

        if external_static_pressure is None:
            external_static_pressure = ducted_external_static_pressure

        ducted_airflow_ratio = self.airflow_ratio(speed_setting, 0, ducted_external_static_pressure)
        ductless_airflow_ratio = self.airflow_ratio(speed_setting, 0, ductless_external_static_pressure)

        ducted_efficacy = self.DUCTED_DESIGN_EFFICACY * ducted_airflow_ratio**1.75

        ductless_efficacy = self.DUCTLESS_DESIGN_EFFICACY * ductless_airflow_ratio * ductless_airflow_ratio

        return (
            ductless_efficacy
            + (ducted_efficacy - ductless_efficacy) * external_static_pressure / ducted_external_static_pressure
        )


# TODO: class ECMTorqueFan(Fan)

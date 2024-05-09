from enum import Enum
import math
import uuid
import datetime
from random import Random
from typing import Union, List, Callable
from pathlib import Path
from dataclasses import dataclass

from scipy import optimize
from numpy import linspace

from koozie import fr_u, to_u

from dimes import (
    DimensionalData,
    DimensionalPlot,
    DisplayData,
    LineProperties,
    sample_colorscale,
)

from .psychrometrics import PsychState, psychrolib, STANDARD_CONDITIONS
from .conditions import HeatingConditions, CoolingConditions, OperatingConditions
from .defrost import Defrost, DefrostControl
from .util import find_nearest, limit_check
from .models import RESNETDXModel, DXModel
from .fan import Fan
from .enums import StagingType


def interpolate_separate(f, f2, cond_1, cond_2, x):
    return f(cond_1) + (f2(cond_2) - f(cond_1)) / (
        cond_2.outdoor.db - cond_1.outdoor.db
    ) * (x - cond_1.outdoor.db)


def interpolate(f, cond_1, cond_2, x):
    return interpolate_separate(f, f, cond_1, cond_2, x)


class CyclingMethod(Enum):
    BETWEEN_LOW_FULL = 1
    BETWEEN_OFF_FULL = 2


class AHRIVersion(Enum):
    AHRI_210_240_2017 = 1
    AHRI_210_240_2023 = 2


# AHRI 210/240 2017 distributions
class HeatingDistribution:
    outdoor_drybulbs = [
        fr_u(62.0 - delta * 5.0, "°F") for delta in range(18)
    ]  # 62.0 to -23 F by 5 F increments

    def __init__(
        self,
        outdoor_design_temperature: float = fr_u(5.0, "°F"),
        # fmt: off
        fractional_hours: List[float]=[0.132, 0.111, 0.103, 0.093, 0.100, 0.109, 0.126, 0.087, 0.055,0.036, 0.026, 0.013, 0.006, 0.002, 0.001, 0, 0, 0],
        # fmt: on
        c=None,
        c_vs=None,
        zero_load_temperature=None,
    ) -> None:
        self.outdoor_design_temperature = outdoor_design_temperature
        self.c = c
        self.c_vs = c_vs
        self.zero_load_temperature = zero_load_temperature
        hour_fraction_sum = sum(fractional_hours)
        if hour_fraction_sum < 0.98 or hour_fraction_sum > 1.02:
            # Issue with 2023 standard, unsure how to interpret
            # print(f"Warning: HeatingDistribution sum of fractional hours ({hour_fraction_sum}) is not 1.0.")
            # print(f"         Values will be re-normalized.")
            self.fractional_hours = [n / hour_fraction_sum for n in fractional_hours]
        else:
            self.fractional_hours = fractional_hours
        self.number_of_bins = len(self.fractional_hours)
        if self.number_of_bins != 18:
            raise Exception("Heating distributions must be provided in 18 bins.")


class CoolingDistribution:
    outdoor_drybulbs = [
        fr_u(67.0 + delta * 5.0, "°F") for delta in range(8)
    ]  # 67.0 to 102 F by 5 F increments
    fractional_hours = [0.214, 0.231, 0.216, 0.161, 0.104, 0.052, 0.018, 0.004]

    def __init__(self):
        self.number_of_bins = len(self.fractional_hours)


class DXUnitMetadata:
    def __init__(
        self,
        description="",
        data_source="https://github.com/bigladder/resdx",
        notes="",
        compressor_type="",
        uuid_seed=None,
        data_version=1,
    ):
        self.description = description
        self.data_source = data_source
        self.notes = notes
        self.compressor_type = compressor_type
        self.uuid_seed = uuid_seed
        self.data_version = data_version


class DXUnit:
    # fmt: off
    regional_heating_distributions = {
        AHRIVersion.AHRI_210_240_2017: {
            1: HeatingDistribution(fr_u(37.0,"°F"), [0.291,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
            2: HeatingDistribution(fr_u(27.0,"°F"), [0.215,0.189,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
            3: HeatingDistribution(fr_u(17.0,"°F"), [0.153,0.142,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000]),
            4: HeatingDistribution(fr_u(5.0,"°F"),  [0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0.000,0.000,0.000]),
            5: HeatingDistribution(fr_u(-10.0,"°F"),[0.106,0.092,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001]),
            6: HeatingDistribution(fr_u(30.0,"°F"), [0.113,0.206,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000])},
        AHRIVersion.AHRI_210_240_2023: {
            # Note: AHRI 2023 issue: None of these distributions add to 1.0!
            1: HeatingDistribution(fr_u(37.0,"°F"), [0.000,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.10, 1.03, fr_u(58.0,"°F")),
            2: HeatingDistribution(fr_u(27.0,"°F"), [0.000,0.000,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.06, 0.99, fr_u(57.0,"°F")),
            3: HeatingDistribution(fr_u(17.0,"°F"), [0.000,0.000,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000], 1.30, 1.21, fr_u(56.0,"°F")),
            4: HeatingDistribution(fr_u(5.0,"°F"),  [0.000,0.000,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0.000,0.000,0.000], 1.15, 1.07, fr_u(55.0,"°F")),
            5: HeatingDistribution(fr_u(-10.0,"°F"),[0.000,0.000,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001], 1.16, 1.08, fr_u(55.0,"°F")),
            6: HeatingDistribution(fr_u(30.0,"°F"), [0.000,0.000,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.11, 1.03, fr_u(57.0,"°F"))},
    }
    # fmt: on

    cooling_distribution = CoolingDistribution()

    standard_design_heating_requirements = [
        (fr_u(5000, "Btu/h") + i * fr_u(5000, "Btu/h")) for i in range(0, 8)
    ] + [(fr_u(50000, "Btu/h") + i * fr_u(10000, "Btu/h")) for i in range(0, 9)]

    def __init__(
        # defaults of None are defaulted within this function based on other argument values
        self,
        model: Union[DXModel, None] = None,
        staging_type=None,  # Allow default based on inputs
        # Cooling (rating = AHRI A conditions)
        rated_net_total_cooling_capacity=fr_u(3.0, "ton_ref"),
        rated_gross_cooling_cop=3.72,
        rated_net_cooling_cop=None,
        rated_cooling_airflow_per_rated_net_capacity=None,
        c_d_cooling=None,
        cooling_off_temperature=fr_u(125.0, "°F"),
        # Heating (rating = AHRI H1 conditions)
        rated_net_heating_capacity=None,
        rated_gross_heating_cop=3.82,
        rated_net_heating_cop=None,
        rated_heating_airflow_per_rated_net_capacity=None,
        c_d_heating=None,
        heating_off_temperature=fr_u(0.0, "°F"),
        heating_on_temperature=None,  # default to heating_off_temperature
        defrost=None,
        # Fan
        fan=None,
        heating_fan_speed=None,  # Map of cooling compressor speed to fan speed setting
        cooling_fan_speed=None,  # Map of heating compressor speed to fan speed setting
        rated_heating_fan_speed=None,  # Map of cooling compressor speed to fan speed setting used for ratings
        rated_cooling_fan_speed=None,  # Map of heating compressor speed to fan speed setting used for ratings
        # Standby
        crankcase_heater_capacity=None,
        crankcase_heater_setpoint_temperature=fr_u(50.0, "°F"),
        # Faults
        refrigerant_charge_deviation=0.0,
        # Ratings
        cycling_method=CyclingMethod.BETWEEN_LOW_FULL,
        cooling_full_load_speed=0,  # The first entry (index = 0) in arrays reflects AHRI "full" speed.
        cooling_intermediate_speed=None,
        heating_full_load_speed=0,  # The first entry (index = 0) in arrays reflects AHRI "full" speed.
        heating_intermediate_speed=None,
        is_ducted=True,
        rating_standard=AHRIVersion.AHRI_210_240_2023,
        # Used for comparisons and to inform some defaults
        input_seer=None,  # SEER value input (may not match calculated SEER of the model)
        input_eer=None,  # EER value input (may not match calculated EER of the model)
        input_hspf=None,  # HSPF value input (may not match calculated HSPF of the model)
        metadata=None,
        **kwargs,
    ):  # Additional inputs used for specific models
        # Initialize direct values
        self.kwargs = kwargs

        self.model: DXModel = RESNETDXModel() if model is None else model

        self.metadata = DXUnitMetadata() if metadata is None else metadata

        self.model.set_system(self)

        # Inputs used by some models
        # Ratings
        self.input_seer = input_seer
        self.input_hspf = input_hspf
        self.input_eer = input_eer

        self.input_rated_net_heating_cop = rated_net_heating_cop

        self.cycling_method = cycling_method
        self.is_ducted = is_ducted

        self.cooling_off_temperature = cooling_off_temperature

        if defrost is None:
            self.defrost = Defrost()
        else:
            self.defrost = defrost

        self.crankcase_heater_setpoint_temperature = (
            crankcase_heater_setpoint_temperature
        )
        self.rating_standard = rating_standard
        self.heating_off_temperature = heating_off_temperature
        self.refrigerant_charge_deviation = refrigerant_charge_deviation
        if heating_on_temperature == None:
            self.heating_on_temperature = self.heating_off_temperature

        # Placeholder for additional data set specific to the model
        self.model_data: dict = {}

        # Number of stages/speeds
        if type(rated_net_total_cooling_capacity) is list:
            self.number_of_cooling_speeds = len(rated_net_total_cooling_capacity)
        elif staging_type is not None:
            self.number_of_cooling_speeds = (
                staging_type.value if staging_type != StagingType.VARIABLE_SPEED else 4
            )
        else:
            self.number_of_cooling_speeds = 1

        if type(rated_net_heating_capacity) is list:
            self.number_of_heating_speeds = len(rated_net_heating_capacity)
        elif staging_type is not None:
            self.number_of_heating_speeds = (
                staging_type.value if staging_type != StagingType.VARIABLE_SPEED else 4
            )
        else:
            self.number_of_heating_speeds = 1

        self.cooling_full_load_speed = cooling_full_load_speed
        self.heating_full_load_speed = heating_full_load_speed

        self.cooling_boost_speed: Union[int, None] = (
            0 if self.cooling_full_load_speed > 0 else None
        )

        self.heating_boost_speed: Union[int, None] = (
            0 if self.heating_full_load_speed > 0 else None
        )

        if cooling_intermediate_speed is None:
            self.cooling_intermediate_speed = cooling_full_load_speed + 1
        else:
            self.cooling_intermediate_speed = cooling_intermediate_speed

        if heating_intermediate_speed is None:
            self.heating_intermediate_speed = heating_full_load_speed + 1
        else:
            self.heating_intermediate_speed = heating_intermediate_speed

        if self.number_of_cooling_speeds > 1:
            self.cooling_low_speed = self.number_of_cooling_speeds - 1
        else:
            self.cooling_low_speed = None

        if self.number_of_heating_speeds > 1:
            self.heating_low_speed = self.number_of_heating_speeds - 1
        else:
            self.heating_low_speed = None

        if staging_type is None:
            self.staging_type = StagingType(min(self.number_of_heating_speeds, 3))
        else:
            self.staging_type = staging_type

        # Placeholders for derived staging array values
        self.set_placeholder_arrays()

        # Default fan maps (if None, will be set in set_net_capacities_and_fan)
        self.heating_fan_speed = heating_fan_speed
        self.cooling_fan_speed = cooling_fan_speed
        self.rated_heating_fan_speed = rated_heating_fan_speed
        self.rated_cooling_fan_speed = rated_cooling_fan_speed
        self.rated_cooling_airflow_per_rated_net_capacity = (
            rated_cooling_airflow_per_rated_net_capacity
        )
        self.rated_heating_airflow_per_rated_net_capacity = (
            rated_heating_airflow_per_rated_net_capacity
        )

        # Set net capacities and fan
        self.rated_net_total_cooling_capacity: List[float]
        self.rated_net_heating_capacity: List[float]
        self.fan: Fan
        self.model.set_net_capacities_and_fan(
            rated_net_total_cooling_capacity, rated_net_heating_capacity, fan
        )

        # Degradation coefficients
        self.c_d_cooling: float
        self.c_d_heating: float
        self.model.set_c_d_cooling(c_d_cooling)
        self.model.set_c_d_heating(c_d_heating)

        self.rated_full_flow_external_static_pressure = (
            self.get_rated_full_flow_rated_pressure()
        )

        # Derived gross capacities
        for i in range(self.number_of_cooling_speeds):
            self.rated_gross_total_cooling_capacity[i] = (
                self.rated_net_total_cooling_capacity[i]
                + self.rated_cooling_fan_power[i]
            )

        for i in range(self.number_of_heating_speeds):
            self.rated_gross_heating_capacity[i] = (
                self.rated_net_heating_capacity[i] - self.rated_heating_fan_power[i]
            )

        # Crankcase
        if crankcase_heater_capacity is None:
            self.crankcase_heater_capacity = 10.0 * to_u(
                self.rated_net_total_cooling_capacity[self.cooling_full_load_speed],
                "ton_ref",
            )
        else:
            self.crankcase_heater_capacity = crankcase_heater_capacity

        # Set rating conditions
        self.set_rating_conditions()

        # Sensible cooling capacities
        self.set_sensible_cooling_variables()

        # COP determinations
        if rated_net_cooling_cop is None and rated_gross_cooling_cop is None:
            raise RuntimeError(
                "Must define either 'rated_net_cooling_cop' or 'rated_gross_cooling_cop'."
            )

        if rated_net_heating_cop is None and rated_gross_heating_cop is None:
            raise RuntimeError(
                "Must define either 'rated_net_heating_cop' or 'rated_gross_heating_cop'."
            )

        if rated_net_cooling_cop is not None:
            self.model.set_rated_net_cooling_cop(rated_net_cooling_cop)

        if rated_gross_cooling_cop is not None:
            self.model.set_rated_gross_cooling_cop(rated_gross_cooling_cop)

        if rated_net_heating_cop is not None:
            self.model.set_rated_net_heating_cop(rated_net_heating_cop)

        if rated_gross_heating_cop is not None:
            self.model.set_rated_gross_heating_cop(rated_gross_heating_cop)

        ## Check for errors

        # Check to make sure all cooling arrays are the same size if not output a warning message
        self.check_array_lengths()

        # Check to make sure arrays are in descending order
        self.check_array_order(self.rated_net_total_cooling_capacity)
        self.check_array_order(self.rated_net_heating_capacity)

    def set_rating_conditions(self) -> None:
        self.rated_full_flow_external_static_pressure = (
            self.get_rated_full_flow_rated_pressure()
        )

        self.A_full_cond: CoolingConditions = self.make_condition(
            CoolingConditions, compressor_speed=self.cooling_full_load_speed
        )
        self.B_full_cond: CoolingConditions = self.make_condition(
            CoolingConditions,
            outdoor=PsychState(drybulb=fr_u(82.0, "°F"), wetbulb=fr_u(65.0, "°F")),
            compressor_speed=self.cooling_full_load_speed,
        )

        self.H1_full_cond: HeatingConditions = self.make_condition(
            HeatingConditions, compressor_speed=self.heating_full_load_speed
        )
        self.H2_full_cond = self.make_condition(
            HeatingConditions,
            outdoor=PsychState(drybulb=fr_u(35.0, "°F"), wetbulb=fr_u(33.0, "°F")),
            compressor_speed=self.heating_full_load_speed,
        )
        self.H3_full_cond: HeatingConditions = self.make_condition(
            HeatingConditions,
            outdoor=PsychState(drybulb=fr_u(17.0, "°F"), wetbulb=fr_u(15.0, "°F")),
            compressor_speed=self.heating_full_load_speed,
        )
        self.H4_full_cond: HeatingConditions = self.make_condition(
            HeatingConditions,
            outdoor=PsychState(drybulb=fr_u(5.0, "°F"), wetbulb=fr_u(3.0, "°F")),
            compressor_speed=self.heating_full_load_speed,
        )

        if self.staging_type != StagingType.SINGLE_STAGE:
            if self.staging_type == StagingType.VARIABLE_SPEED:
                self.A_int_cond: CoolingConditions = self.make_condition(
                    CoolingConditions, compressor_speed=self.cooling_intermediate_speed
                )  # Not used in AHRI ratings, only used for 'rated' SHR calculations at low speeds
                self.E_int_cond: CoolingConditions = self.make_condition(
                    CoolingConditions,
                    outdoor=PsychState(
                        drybulb=fr_u(87.0, "°F"), wetbulb=fr_u(69.0, "°F")
                    ),
                    compressor_speed=self.cooling_intermediate_speed,
                )

                self.H2_int_cond: HeatingConditions = self.make_condition(
                    HeatingConditions,
                    outdoor=PsychState(
                        drybulb=fr_u(35.0, "°F"), wetbulb=fr_u(33.0, "°F")
                    ),
                    compressor_speed=self.heating_intermediate_speed,
                )

            if self.cooling_boost_speed is not None:
                self.A_boost_cond: CoolingConditions = self.make_condition(
                    CoolingConditions, compressor_speed=self.cooling_boost_speed
                )  # TODO: Evaluate impacts on AHRI ratings

            self.A_low_cond: CoolingConditions = self.make_condition(
                CoolingConditions, compressor_speed=self.cooling_low_speed
            )  # Not used in AHRI ratings, only used for 'rated' SHR calculations at low speeds
            self.B_low_cond: CoolingConditions = self.make_condition(
                CoolingConditions,
                outdoor=PsychState(drybulb=fr_u(82.0, "°F"), wetbulb=fr_u(65.0, "°F")),
                compressor_speed=self.cooling_low_speed,
            )
            self.F_low_cond: CoolingConditions = self.make_condition(
                CoolingConditions,
                outdoor=PsychState(drybulb=fr_u(67.0, "°F"), wetbulb=fr_u(53.5, "°F")),
                compressor_speed=self.cooling_low_speed,
            )

            self.H0_low_cond: HeatingConditions = self.make_condition(
                HeatingConditions,
                outdoor=PsychState(drybulb=fr_u(62.0, "°F"), wetbulb=fr_u(56.5, "°F")),
                compressor_speed=self.heating_low_speed,
            )
            self.H1_low_cond: HeatingConditions = self.make_condition(
                HeatingConditions, compressor_speed=self.heating_low_speed
            )
            self.H2_low_cond: HeatingConditions = self.make_condition(
                HeatingConditions,
                outdoor=PsychState(drybulb=fr_u(35.0, "°F"), wetbulb=fr_u(33.0, "°F")),
                compressor_speed=self.heating_low_speed,
            )
            self.H3_low_cond: HeatingConditions = self.make_condition(
                HeatingConditions,
                outdoor=PsychState(drybulb=fr_u(17.0, "°F"), wetbulb=fr_u(15.0, "°F")),
                compressor_speed=self.heating_low_speed,
            )

    def set_sensible_cooling_variables(self):
        self.rated_gross_shr_cooling[self.cooling_full_load_speed] = (
            self.model.gross_shr(self.A_full_cond)
        )
        self.calculate_rated_bypass_factor(self.A_full_cond)

        if self.staging_type != StagingType.SINGLE_STAGE:
            if self.staging_type == StagingType.VARIABLE_SPEED:
                self.rated_gross_shr_cooling[self.cooling_intermediate_speed] = (
                    self.model.gross_shr(self.A_int_cond)
                )
                self.calculate_rated_bypass_factor(self.A_int_cond)

                if self.cooling_boost_speed is not None:
                    self.rated_gross_shr_cooling[self.cooling_boost_speed] = (
                        self.model.gross_shr(self.A_boost_cond)
                    )
                    self.calculate_rated_bypass_factor(self.A_boost_cond)

            self.rated_gross_shr_cooling[self.cooling_low_speed] = self.model.gross_shr(
                self.A_low_cond
            )
            self.calculate_rated_bypass_factor(self.A_low_cond)

    def reset_rated_flow_rates(
        self,
        rated_cooling_airflow_per_rated_net_capacity,
        rated_heating_airflow_per_rated_net_capacity,
    ):
        # Make new fan settings
        if type(rated_cooling_airflow_per_rated_net_capacity) is not list:
            rated_cooling_airflow_per_rated_net_capacity = [
                rated_cooling_airflow_per_rated_net_capacity
            ] * self.number_of_cooling_speeds

        if type(rated_heating_airflow_per_rated_net_capacity) is not list:
            rated_heating_airflow_per_rated_net_capacity = [
                rated_heating_airflow_per_rated_net_capacity
            ] * self.number_of_heating_speeds

        full_airflow = (
            self.rated_net_total_cooling_capacity[0]
            * rated_cooling_airflow_per_rated_net_capacity[0]
        )
        for i, airflow_per_capacity in enumerate(
            rated_cooling_airflow_per_rated_net_capacity
        ):
            airflow = self.rated_net_total_cooling_capacity[i] * airflow_per_capacity
            pressure = self.calculate_rated_pressure(airflow, full_airflow)
            self.fan.add_speed(airflow, pressure)
            new_fan_speed = self.fan.number_of_speeds - 1

            if i == self.cooling_full_load_speed:
                self.A_full_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.B_full_cond.set_new_fan_speed(new_fan_speed, airflow)
            if i == self.cooling_intermediate_speed:
                self.A_int_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.E_int_cond.set_new_fan_speed(new_fan_speed, airflow)
            if i == self.cooling_low_speed:
                self.A_low_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.B_low_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.F_low_cond.set_new_fan_speed(new_fan_speed, airflow)

        full_airflow = (
            self.rated_net_heating_capacity[0]
            * rated_heating_airflow_per_rated_net_capacity[0]
        )
        for i, airflow_per_capacity in enumerate(
            rated_heating_airflow_per_rated_net_capacity
        ):
            airflow = self.rated_net_heating_capacity[i] * airflow_per_capacity
            pressure = self.calculate_rated_pressure(airflow, full_airflow)
            self.fan.add_speed(airflow, pressure)
            new_fan_speed = self.fan.number_of_speeds - 1

            if i == self.heating_full_load_speed:
                self.H1_full_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H2_full_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H3_full_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H4_full_cond.set_new_fan_speed(new_fan_speed, airflow)
            if i == self.heating_intermediate_speed:
                self.H2_int_cond.set_new_fan_speed(new_fan_speed, airflow)
            if i == self.heating_low_speed:
                self.H0_low_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H1_low_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H2_low_cond.set_new_fan_speed(new_fan_speed, airflow)
                self.H3_low_cond.set_new_fan_speed(new_fan_speed, airflow)

    def check_array_length(self, array, expected_length):
        if len(array) != expected_length:
            raise RuntimeError(
                f"Unexpected array length ({len(array)}). Number of speeds is {expected_length}. Array items are {array}."
            )

    def check_array_lengths(self):
        self.check_array_length(
            self.rated_net_total_cooling_capacity, self.number_of_cooling_speeds
        )
        self.check_array_length(
            self.rated_gross_shr_cooling, self.number_of_cooling_speeds
        )
        self.check_array_length(
            self.rated_gross_heating_cop, self.number_of_heating_speeds
        )
        self.check_array_length(
            self.rated_net_heating_capacity, self.number_of_heating_speeds
        )

    def check_array_order(self, array):
        if not all(earlier >= later for earlier, later in zip(array, array[1:])):
            raise RuntimeError(
                f"Arrays must be in order of decreasing capacity. Array items are {array}."
            )

    def set_placeholder_arrays(self):
        self.rated_cooling_fan_power = [None] * self.number_of_cooling_speeds
        self.rated_gross_total_cooling_capacity = [None] * self.number_of_cooling_speeds
        self.rated_gross_shr_cooling = [None] * self.number_of_cooling_speeds
        self.rated_bypass_factor = [None] * self.number_of_cooling_speeds
        self.normalized_ntu = [None] * self.number_of_cooling_speeds
        self.rated_net_cooling_cop = [None] * self.number_of_cooling_speeds
        self.rated_net_cooling_power = [None] * self.number_of_cooling_speeds
        self.rated_gross_cooling_power = [None] * self.number_of_cooling_speeds
        self.rated_gross_cooling_cop = [None] * self.number_of_cooling_speeds
        self.rated_cooling_airflow = [None] * self.number_of_cooling_speeds
        self.rated_cooling_external_static_pressure = [
            None
        ] * self.number_of_cooling_speeds

        self.rated_heating_fan_power = [None] * self.number_of_heating_speeds
        self.rated_gross_heating_capacity = [None] * self.number_of_heating_speeds
        self.rated_net_heating_cop = [None] * self.number_of_heating_speeds
        self.rated_net_heating_power = [None] * self.number_of_heating_speeds
        self.rated_gross_heating_power = [None] * self.number_of_heating_speeds
        self.rated_gross_heating_cop = [None] * self.number_of_heating_speeds
        self.rated_heating_airflow = [None] * self.number_of_heating_speeds
        self.rated_heating_external_static_pressure = [
            None
        ] * self.number_of_heating_speeds

    def make_condition(
        self, condition_type, compressor_speed=0, indoor=None, outdoor=None
    ):
        if indoor is None:
            indoor = condition_type().indoor
        if outdoor is None:
            outdoor = condition_type().outdoor

        if condition_type == CoolingConditions:
            rated_net_capacity = self.rated_net_total_cooling_capacity[compressor_speed]
            rated_airflow = self.rated_cooling_airflow[compressor_speed]
            rated_full_airflow = self.rated_cooling_airflow[0]
            rated_fan_speed = self.rated_cooling_fan_speed[compressor_speed]
        else:  # if condition_type == HeatingConditions:
            rated_net_capacity = self.rated_net_heating_capacity[compressor_speed]
            rated_airflow = self.rated_heating_airflow[compressor_speed]
            rated_full_airflow = self.rated_heating_airflow[0]
            rated_fan_speed = self.rated_heating_fan_speed[compressor_speed]

        rated_flow_external_static_pressure = self.calculate_rated_pressure(
            rated_airflow, rated_full_airflow
        )
        condition = condition_type(
            indoor=indoor,
            outdoor=outdoor,
            compressor_speed=compressor_speed,
            fan_speed=rated_fan_speed,
            rated_flow_external_static_pressure=rated_flow_external_static_pressure,
        )
        condition.set_rated_volumetric_airflow(rated_airflow, rated_net_capacity)
        return condition

    def get_rated_full_flow_rated_pressure(self):
        if not self.is_ducted:
            return 0.0
        if self.rating_standard == AHRIVersion.AHRI_210_240_2017:
            # TODO: Add Small-duct, High-velocity Systems
            if self.rated_net_total_cooling_capacity[0] <= fr_u(29000, "Btu/h"):
                return fr_u(0.1, "in_H2O")
            elif self.rated_net_total_cooling_capacity[0] <= fr_u(43000, "Btu/h"):
                return fr_u(0.15, "in_H2O")
            else:
                return fr_u(0.2, "in_H2O")
        elif self.rating_standard == AHRIVersion.AHRI_210_240_2023:
            # TODO: Add exceptional system types
            return fr_u(0.5, "in_H2O")

    def calculate_rated_pressure(self, rated_airflow, rated_full_airflow):
        return (
            self.rated_full_flow_external_static_pressure
            * (rated_airflow / rated_full_airflow) ** 2
        )

    def set_rating_standard(self, rating_standard):
        self.rating_standard = rating_standard
        # Reset rating conditions to use the any differences between standards
        self.set_rating_conditions()

    ### For cooling ###
    def cooling_fan_power(self, conditions=None):
        if conditions is None:
            conditions = self.A_full_cond
        return self.fan.power(conditions.fan_speed, conditions.external_static_pressure)

    def cooling_fan_heat(self, conditions):
        return self.cooling_fan_power(conditions)

    def gross_total_cooling_capacity(self, conditions=None):
        if conditions is None:
            conditions = self.A_full_cond
        return limit_check(
            self.model.gross_total_cooling_capacity(conditions)
            * self.model.gross_total_cooling_capacity_charge_factor(conditions),
            min=0.0,
        )

    def gross_sensible_cooling_capacity(self, conditions=None):
        if conditions is None:
            conditions = self.A_full_cond
        return limit_check(
            self.model.gross_sensible_cooling_capacity(conditions), min=0.0
        )

    def gross_shr(self, conditions=None):
        return limit_check(
            self.gross_sensible_cooling_capacity(conditions)
            / self.gross_total_cooling_capacity(conditions),
            min=0.0,
            max=1.0,
        )

    def gross_cooling_power(self, conditions=None):
        if conditions is None:
            conditions = self.A_full_cond
        return limit_check(
            self.model.gross_cooling_power(conditions)
            * self.model.gross_cooling_power_charge_factor(conditions),
            min=0.0,
        )

    def net_total_cooling_capacity(self, conditions=None):
        return self.gross_total_cooling_capacity(conditions) - self.cooling_fan_heat(
            conditions
        )

    def net_sensible_cooling_capacity(self, conditions=None):
        return self.gross_sensible_cooling_capacity(conditions) - self.cooling_fan_heat(
            conditions
        )

    def net_shr(self, conditions=None):
        return self.net_sensible_cooling_capacity(
            conditions
        ) / self.net_total_cooling_capacity(conditions)

    def net_cooling_power(self, conditions=None):
        return self.gross_cooling_power(conditions) + self.cooling_fan_power(conditions)

    def gross_total_cooling_cop(self, conditions=None):
        return self.gross_total_cooling_capacity(conditions) / self.gross_cooling_power(
            conditions
        )

    def gross_sensible_cooling_cop(self, conditions=None):
        return self.gross_sensible_cooling_capacity(
            conditions
        ) / self.gross_cooling_power(conditions)

    def net_total_cooling_cop(self, conditions=None):
        return self.net_total_cooling_capacity(conditions) / self.net_cooling_power(
            conditions
        )

    def net_sensible_cooling_cop(self, conditions=None):
        return self.net_sensible_cooling_capacity(conditions) / self.net_cooling_power(
            conditions
        )

    def gross_cooling_outlet_state(self, conditions=None, gross_sensible_capacity=None):
        if conditions is None:
            conditions = self.A_full_cond
        if gross_sensible_capacity is None:
            gross_sensible_capacity = self.gross_sensible_cooling_capacity(conditions)

        T_idb = conditions.indoor.db
        h_i = conditions.indoor.get_h()
        m_dot_rated = conditions.mass_airflow
        h_o = h_i - self.gross_total_cooling_capacity(conditions) / m_dot_rated
        T_odb = T_idb - gross_sensible_capacity / (m_dot_rated * conditions.indoor.C_p)
        return PsychState(T_odb, pressure=conditions.indoor.p, enthalpy=h_o)

    def calculate_adp_state(self, inlet_state, outlet_state):
        T_idb = inlet_state.db_C
        w_i = inlet_state.get_hr()
        T_odb = outlet_state.db_C
        w_o = outlet_state.get_hr()

        def root_function(T_ADP):
            return psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, inlet_state.p) - (
                w_i - (w_i - w_o) / (T_idb - T_odb) * (T_idb - T_ADP)
            )

        T_ADP = optimize.newton(root_function, T_idb)
        w_ADP = w_i - (w_i - w_o) / (T_idb - T_odb) * (T_idb - T_ADP)
        # Output an error if ADP calculation method is not applicable:
        if T_odb < T_ADP or w_o < w_ADP:
            raise Exception(
                f"Invalid Apparatus Dew Point (ADP). The rated Sensible Heat Ratio (SHR) might not be valid."
            )
        return PsychState(fr_u(T_ADP, "°C"), pressure=inlet_state.p, hum_rat=w_ADP)

    def calculate_rated_bypass_factor(
        self, conditions: CoolingConditions
    ):  # for rated flow rate
        Q_s_rated = self.rated_gross_shr_cooling[
            conditions.compressor_speed
        ] * self.gross_total_cooling_capacity(conditions)
        outlet_state = self.gross_cooling_outlet_state(
            conditions, gross_sensible_capacity=Q_s_rated
        )
        ADP_state = self.calculate_adp_state(conditions.indoor, outlet_state)
        h_i = conditions.indoor.get_h()
        h_o = outlet_state.get_h()
        h_ADP = ADP_state.get_h()
        self.rated_bypass_factor[conditions.compressor_speed] = (h_o - h_ADP) / (
            h_i - h_ADP
        )
        self.normalized_ntu[conditions.compressor_speed] = (
            -conditions.mass_airflow
            * math.log(self.rated_bypass_factor[conditions.compressor_speed])
        )  # A0 = - m_dot * ln(BF)

    def bypass_factor(self, conditions=None):
        if conditions is None:
            conditions = self.A_full_cond
        return math.exp(
            -self.normalized_ntu[conditions.compressor_speed] / conditions.mass_airflow
        )

    def adp_state(self, conditions=None):
        outlet_state = self.gross_cooling_outlet_state(conditions)
        return self.calculate_adp_state(conditions.indoor, outlet_state)

    def eer(self, conditions=None):
        return to_u(self.net_total_cooling_cop(conditions), "Btu/Wh")

    def seer(self):
        """Based on AHRI 210/240 2023 (unless otherwise noted)"""
        if self.staging_type == StagingType.SINGLE_STAGE:
            plf = 1.0 - 0.5 * self.c_d_cooling  # eq. 11.56
            seer = plf * self.net_total_cooling_cop(
                self.B_full_cond
            )  # eq. 11.55 (using COP to keep things in SI units for now)
        else:  # if self.staging_type == StagingType.TWO_STAGE or self.staging_type == StagingType.VARIABLE_SPEED:
            sizing_factor = 1.1  # eq. 11.61
            q_sum = 0.0
            e_sum = 0.0
            if self.staging_type == StagingType.VARIABLE_SPEED:
                # Intermediate capacity
                q_A_full = self.net_total_cooling_capacity(self.A_full_cond)
                q_B_full = self.net_total_cooling_capacity(self.B_full_cond)
                q_B_low = self.net_total_cooling_capacity(self.B_low_cond)
                q_F_low = self.net_total_cooling_capacity(self.F_low_cond)
                q_E_int = self.net_total_cooling_capacity(self.E_int_cond)
                q_87_low = interpolate(
                    self.net_total_cooling_capacity,
                    self.F_low_cond,
                    self.B_low_cond,
                    fr_u(87.0, "°F"),
                )
                q_87_full = interpolate(
                    self.net_total_cooling_capacity,
                    self.B_full_cond,
                    self.A_full_cond,
                    fr_u(87.0, "°F"),
                )
                N_Cq = (q_E_int - q_87_low) / (q_87_full - q_87_low)
                M_Cq = (q_B_low - q_F_low) / (fr_u(82, "°F") - fr_u(67.0, "°F")) * (
                    1.0 - N_Cq
                ) + (q_A_full - q_B_full) / (fr_u(95, "°F") - fr_u(82.0, "°F")) * N_Cq

                # Intermediate power
                p_A_full = self.net_cooling_power(self.A_full_cond)
                p_B_full = self.net_cooling_power(self.B_full_cond)
                p_B_low = self.net_cooling_power(self.B_low_cond)
                p_F_low = self.net_cooling_power(self.F_low_cond)
                p_E_int = self.net_cooling_power(self.E_int_cond)
                p_87_low = interpolate(
                    self.net_cooling_power,
                    self.F_low_cond,
                    self.B_low_cond,
                    fr_u(87.0, "°F"),
                )
                p_87_full = interpolate(
                    self.net_cooling_power,
                    self.B_full_cond,
                    self.A_full_cond,
                    fr_u(87.0, "°F"),
                )
                N_CE = (p_E_int - p_87_low) / (p_87_full - p_87_low)
                M_CE = (p_B_low - p_F_low) / (fr_u(82, "°F") - fr_u(67.0, "°F")) * (
                    1.0 - N_CE
                ) + (p_A_full - p_B_full) / (fr_u(95, "°F") - fr_u(82.0, "°F")) * N_CE

            for i in range(self.cooling_distribution.number_of_bins):
                t = self.cooling_distribution.outdoor_drybulbs[i]
                n = self.cooling_distribution.fractional_hours[i]
                bl = max(
                    (
                        (t - fr_u(65.0, "°F"))
                        / (fr_u(95, "°F") - fr_u(65.0, "°F"))
                        * self.net_total_cooling_capacity(self.A_full_cond)
                        / sizing_factor,
                        0.0,
                    )
                )  # eq. 11.60
                q_low = interpolate(
                    self.net_total_cooling_capacity, self.F_low_cond, self.B_low_cond, t
                )  # eq. 11.62
                p_low = interpolate(
                    self.net_cooling_power, self.F_low_cond, self.B_low_cond, t
                )  # eq. 11.63
                q_full = interpolate(
                    self.net_total_cooling_capacity,
                    self.B_full_cond,
                    self.A_full_cond,
                    t,
                )  # eq. 11.64
                p_full = interpolate(
                    self.net_cooling_power, self.B_full_cond, self.A_full_cond, t
                )  # eq. 11.65
                if self.staging_type == StagingType.TWO_STAGE:
                    if bl <= q_low:
                        clf_low = bl / q_low  # eq. 11.68
                        plf_low = 1.0 - self.c_d_cooling * (1.0 - clf_low)  # eq. 11.69
                        q = clf_low * q_low * n  # eq. 11.66
                        e = clf_low * p_low * n / plf_low  # eq. 11.67
                    elif (
                        bl < q_full
                        and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL
                    ):
                        clf_low = (q_full - bl) / (q_full - q_low)  # eq. 11.74
                        clf_full = 1.0 - clf_low  # eq. 11.75
                        q = (clf_low * q_low + clf_full * q_full) * n  # eq. 11.72
                        e = (clf_low * p_low + clf_full * p_full) * n  # eq. 11.73
                    elif (
                        bl < q_full
                        and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL
                    ):
                        clf_full = bl / q_full  # eq. 11.78
                        plf_full = 1.0 - self.c_d_cooling * (
                            1.0 - clf_full
                        )  # eq. 11.79
                        q = clf_full * q_full * n  # eq. 11.76
                        e = clf_full * p_full * n / plf_full  # eq. 11.77
                    else:  # elif bl >= q_full
                        q = q_full * n
                        e = p_full * n
                else:  # if self.staging_type == StagingType.VARIABLE_SPEED
                    q_int = q_E_int + M_Cq * (t - (fr_u(87, "°F")))
                    p_int = p_E_int + M_CE * (t - (fr_u(87, "°F")))
                    cop_low = q_low / p_low
                    cop_int = q_int / p_int
                    cop_full = q_full / p_full

                    if bl <= q_low:
                        clf_low = bl / q_low  # eq. 11.68
                        plf_low = 1.0 - self.c_d_cooling * (1.0 - clf_low)  # eq. 11.69
                        q = clf_low * q_low * n  # eq. 11.66
                        e = clf_low * p_low * n / plf_low  # eq. 11.67
                    elif bl < q_int:
                        cop_int_bin = cop_low + (cop_int - cop_low) / (
                            q_int - q_low
                        ) * (
                            bl - q_low
                        )  # eq. 11.101 (2023)
                        q = bl * n
                        e = q / cop_int_bin
                    elif bl <= q_full:
                        cop_int_bin = cop_int + (cop_full - cop_int) / (
                            q_full - q_int
                        ) * (
                            bl - q_int
                        )  # eq. 11.101 (2023)
                        q = bl * n
                        e = q / cop_int_bin
                    else:  # elif bl >= q_full
                        q = q_full * n
                        e = p_full * n

                q_sum += q
                e_sum += e

            seer = q_sum / e_sum  # e.q. 11.59
        return to_u(seer, "Btu/Wh")

    ### For heating ###
    def heating_fan_power(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        return self.fan.power(conditions.fan_speed, conditions.external_static_pressure)

    def heating_fan_heat(self, conditions):
        return self.heating_fan_power(conditions)

    def gross_steady_state_heating_capacity(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        return self.model.gross_steady_state_heating_capacity(
            conditions
        ) * self.model.gross_steady_state_heating_capacity_charge_factor(conditions)

    def gross_steady_state_heating_power(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        return self.model.gross_steady_state_heating_power(
            conditions
        ) * self.model.gross_steady_state_heating_power_charge_factor(conditions)

    def gross_integrated_heating_capacity(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        return self.model.gross_integrated_heating_capacity(conditions)

    def gross_integrated_heating_power(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        return self.model.gross_integrated_heating_power(conditions)

    def net_steady_state_heating_capacity(self, conditions=None):
        return self.gross_steady_state_heating_capacity(
            conditions
        ) + self.heating_fan_heat(conditions)

    def net_steady_state_heating_power(self, conditions=None):
        return self.gross_steady_state_heating_power(
            conditions
        ) + self.heating_fan_power(conditions)

    def net_integrated_heating_capacity(self, conditions=None):
        return self.gross_integrated_heating_capacity(
            conditions
        ) + self.heating_fan_heat(conditions)

    def net_integrated_heating_power(self, conditions=None):
        return self.gross_integrated_heating_power(conditions) + self.heating_fan_power(
            conditions
        )

    def gross_steady_state_heating_cop(self, conditions=None):
        return self.gross_steady_state_heating_capacity(
            conditions
        ) / self.gross_steady_state_heating_power(conditions)

    def gross_integrated_heating_cop(self, conditions=None):
        return self.gross_integrated_heating_capacity(
            conditions
        ) / self.gross_integrated_heating_power(conditions)

    def net_steady_state_heating_cop(self, conditions=None):
        return self.net_steady_state_heating_capacity(
            conditions
        ) / self.net_steady_state_heating_power(conditions)

    def net_integrated_heating_cop(self, conditions=None):
        return self.net_integrated_heating_capacity(
            conditions
        ) / self.net_integrated_heating_power(conditions)

    def gross_heating_capacity_ratio(self, conditions=None):
        return self.gross_integrated_heating_capacity(
            conditions
        ) / self.gross_steady_state_heating_capacity(conditions)

    def net_heating_capacity_ratio(self, conditions=None):
        return self.net_integrated_heating_capacity(
            conditions
        ) / self.net_steady_state_heating_capacity(conditions)

    def gross_heating_power_ratio(self, conditions=None):
        return self.gross_integrated_heating_power(
            conditions
        ) / self.gross_steady_state_heating_power(conditions)

    def net_heating_power_ratio(self, conditions=None):
        return self.net_integrated_heating_power(
            conditions
        ) / self.net_steady_state_heating_power(conditions)

    def gross_heating_output_state(self, conditions=None):
        if conditions is None:
            conditions = self.H1_full_cond
        T_odb = conditions.indoor.db + self.gross_steady_state_heating_capacity(
            conditions
        ) / (conditions.mass_airflow * conditions.indoor.C_p)
        return PsychState(
            T_odb, pressure=conditions.indoor.p, hum_rat=conditions.indoor.get_hr()
        )

    def hspf(self, region=4):
        """Based on AHRI 210/240 2023 (unless otherwise noted)"""
        q_sum = 0.0
        e_sum = 0.0
        rh_sum = 0.0

        heating_distribution = self.regional_heating_distributions[
            self.rating_standard
        ][region]
        t_od = heating_distribution.outdoor_design_temperature

        if self.rating_standard == AHRIVersion.AHRI_210_240_2017:
            c = 0.77  # eq. 11.110 (agreement factor)
            dhr_min = (
                self.net_steady_state_heating_capacity(self.H1_full_cond)
                * (fr_u(65, "°F") - t_od)
                / (fr_u(60, "°R"))
            )  # eq. 11.111
            dhr_min = find_nearest(self.standard_design_heating_requirements, dhr_min)
        else:  # if self.rating_standard == AHRIVersion.AHRI_210_240_2023:
            if self.staging_type == StagingType.VARIABLE_SPEED:
                c_x = heating_distribution.c_vs
            else:
                c_x = heating_distribution.c
            t_zl = heating_distribution.zero_load_temperature
            q_A_full = self.net_total_cooling_capacity(self.A_full_cond)

        if self.staging_type == StagingType.VARIABLE_SPEED:
            # Intermediate capacity
            q_H0_low = self.net_steady_state_heating_capacity(self.H0_low_cond)
            q_H1_low = self.net_steady_state_heating_capacity(self.H1_low_cond)
            q_H2_int = self.net_integrated_heating_capacity(self.H2_int_cond)
            q_H1_full = self.net_steady_state_heating_capacity(self.H1_full_cond)
            q_H2_full = self.net_integrated_heating_capacity(self.H2_full_cond)
            q_H3_full = self.net_steady_state_heating_capacity(self.H3_full_cond)
            q_H4_full = self.net_steady_state_heating_capacity(self.H4_full_cond)
            q_35_low = interpolate(
                self.net_steady_state_heating_capacity,
                self.H0_low_cond,
                self.H1_low_cond,
                fr_u(35.0, "°F"),
            )
            N_Hq = (q_H2_int - q_35_low) / (q_H2_full - q_35_low)
            M_Hq = (q_H0_low - q_H1_low) / (fr_u(62, "°F") - fr_u(47.0, "°F")) * (
                1.0 - N_Hq
            ) + (q_H2_full - q_H3_full) / (fr_u(35, "°F") - fr_u(17.0, "°F")) * N_Hq

            # Intermediate power
            p_H0_low = self.net_steady_state_heating_power(self.H0_low_cond)
            p_H1_low = self.net_steady_state_heating_power(self.H1_low_cond)
            p_H2_int = self.net_integrated_heating_power(self.H2_int_cond)
            p_H1_full = self.net_steady_state_heating_power(self.H1_full_cond)
            p_H2_full = self.net_integrated_heating_power(self.H2_full_cond)
            p_H3_full = self.net_steady_state_heating_power(self.H3_full_cond)
            p_H4_full = self.net_steady_state_heating_power(self.H4_full_cond)
            p_35_low = interpolate(
                self.net_steady_state_heating_power,
                self.H0_low_cond,
                self.H1_low_cond,
                fr_u(35.0, "°F"),
            )
            N_HE = (p_H2_int - p_35_low) / (p_H2_full - p_35_low)
            M_HE = (p_H0_low - p_H1_low) / (fr_u(62, "°F") - fr_u(47.0, "°F")) * (
                1.0 - N_HE
            ) + (p_H2_full - p_H3_full) / (fr_u(35, "°F") - fr_u(17.0, "°F")) * N_HE

        for i in range(heating_distribution.number_of_bins):
            t = heating_distribution.outdoor_drybulbs[i]
            n = heating_distribution.fractional_hours[i]
            if self.rating_standard == AHRIVersion.AHRI_210_240_2017:
                bl = max(
                    (fr_u(65, "°F") - t) / (fr_u(65, "°F") - t_od) * c * dhr_min, 0.0
                )  # eq. 11.109
            else:  # if self.rating_standard == AHRIVersion.AHRI_210_240_2023:
                bl = max((t_zl - t) / (t_zl - t_od) * c_x * q_A_full, 0.0)

            t_ob = fr_u(45, "°F")  # eq. 11.119
            if t >= t_ob or t <= fr_u(17, "°F"):
                q_full = interpolate(
                    self.net_steady_state_heating_capacity,
                    self.H3_full_cond,
                    self.H1_full_cond,
                    t,
                )  # eq. 11.117
                p_full = interpolate(
                    self.net_steady_state_heating_power,
                    self.H3_full_cond,
                    self.H1_full_cond,
                    t,
                )  # eq. 11.117
            else:  # elif t > fr_u(17,"°F") and t < t_ob
                q_full = interpolate_separate(
                    self.net_steady_state_heating_capacity,
                    self.net_integrated_heating_capacity,
                    self.H3_full_cond,
                    self.H2_full_cond,
                    t,
                )  # eq. 11.118
                p_full = interpolate_separate(
                    self.net_steady_state_heating_power,
                    self.net_integrated_heating_power,
                    self.H3_full_cond,
                    self.H2_full_cond,
                    t,
                )  # eq. 11.117
            cop_full = q_full / p_full

            if t <= self.heating_off_temperature or cop_full < 1.0:
                delta_full = 0.0  # eq. 11.125
            elif t > self.heating_on_temperature:
                delta_full = 1.0  # eq. 11.127
            else:
                delta_full = 0.5  # eq. 11.126

            if q_full > bl:
                hlf_full = bl / q_full  # eq. 11.115 & 11.154
            else:
                hlf_full = 1.0  # eq. 11.116

            if self.staging_type == StagingType.SINGLE_STAGE:
                plf_full = 1.0 - self.c_d_heating * (1.0 - hlf_full)  # eq. 11.125
                e = (
                    p_full * hlf_full * delta_full * n / plf_full
                )  # eq. 11.156 (not shown for single stage)
                rh = (bl - q_full * hlf_full * delta_full) * n  # eq. 11.126
            elif self.staging_type == StagingType.TWO_STAGE:
                t_ob = fr_u(40, "°F")  # eq. 11.134
                if t >= t_ob:
                    q_low = interpolate(
                        self.net_steady_state_heating_capacity,
                        self.H0_low_cond,
                        self.H1_low_cond,
                        t,
                    )  # eq. 11.135
                    p_low = interpolate(
                        self.net_steady_state_heating_power,
                        self.H0_low_cond,
                        self.H1_low_cond,
                        t,
                    )  # eq. 11.138
                elif t <= fr_u(17.0, "°F"):
                    q_low = interpolate(
                        self.net_steady_state_heating_capacity,
                        self.H1_low_cond,
                        self.H3_low_cond,
                        t,
                    )  # eq. 11.137
                    p_low = interpolate(
                        self.net_steady_state_heating_power,
                        self.H1_low_cond,
                        self.H3_low_cond,
                        t,
                    )  # eq. 11.140
                else:
                    q_low = interpolate_separate(
                        self.net_integrated_heating_capacity,
                        self.net_steady_state_heating_capacity,
                        self.H2_low_cond,
                        self.H3_low_cond,
                        t,
                    )  # eq. 11.136
                    p_low = interpolate_separate(
                        self.net_integrated_heating_power,
                        self.net_steady_state_heating_power,
                        self.H2_low_cond,
                        self.H3_low_cond,
                        t,
                    )  # eq. 11.139

                cop_low = q_low / p_low
                if bl <= q_low:
                    if t <= self.heating_off_temperature or cop_low < 1.0:
                        delta_low = 0.0  # eq. 11.159
                    elif t > self.heating_on_temperature:
                        delta_low = 1.0  # eq. 11.160
                    else:
                        delta_low = 0.5  # eq. 11.161

                    hlf_low = bl / q_low  # eq. 11.155
                    plf_low = 1.0 - self.c_d_heating * (1.0 - hlf_low)  # eq. 11.156
                    e = p_low * hlf_low * delta_low * n / plf_low  # eq. 11.153
                    rh = bl * (1.0 - delta_low) * n  # eq. 11.154
                elif (
                    bl > q_low
                    and bl < q_full
                    and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL
                ):
                    hlf_low = (q_full - bl) / (q_full - q_low)  # eq. 11.163
                    hlf_full = 1.0 - hlf_low  # eq. 11.164
                    e = (
                        (p_low * hlf_low + p_full * hlf_full) * delta_low * n
                    )  # eq. 11.162
                    rh = bl * (1.0 - delta_low) * n  # eq. 11.154
                elif (
                    bl > q_low
                    and bl < q_full
                    and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL
                ):
                    hlf_low = (q_full - bl) / (q_full - q_low)  # eq. 11.163
                    plf_full = 1.0 - self.c_d_heating * (1.0 - hlf_low)  # eq. 11.166
                    e = p_full * hlf_full * delta_full * n / plf_full  # eq. 11.165
                    rh = bl * (1.0 - delta_low) * n  # eq. 11.142
                else:  # elif bl >= q_full
                    hlf_full = 1.0  # eq. 11.170
                    e = p_full * hlf_full * delta_full * n  # eq. 11.168
                    rh = (bl - q_full * hlf_full * delta_full) * n  # eq. 11.169
            else:  # if self.staging_type == StagingType.VARIABLE_SPEED:
                # Note: this is strange that there is no defrost cut in the low speed and doesn't use H2 or H3 low
                q_low = interpolate(
                    self.net_steady_state_heating_capacity,
                    self.H0_low_cond,
                    self.H1_low_cond,
                    t,
                )  # eq. 11.177
                p_low = interpolate(
                    self.net_steady_state_heating_power,
                    self.H0_low_cond,
                    self.H1_low_cond,
                    t,
                )  # eq. 11.178
                cop_low = q_low / p_low
                q_int = q_H2_int + M_Hq * (t - (fr_u(35, "°F")))
                p_int = p_H2_int + M_HE * (t - (fr_u(35, "°F")))
                cop_int = q_int / p_int

                if bl <= q_low:
                    if t <= self.heating_off_temperature or cop_low < 1.0:
                        delta_low = 0.0  # eq. 11.159
                    elif t > self.heating_on_temperature:
                        delta_low = 1.0  # eq. 11.160
                    else:
                        delta_low = 0.5  # eq. 11.161

                    hlf_low = bl / q_low  # eq. 11.155
                    plf_low = 1.0 - self.c_d_heating * (1.0 - hlf_low)  # eq. 11.156
                    e = p_low * hlf_low * delta_low * n / plf_low  # eq. 11.153
                    rh = bl * (1.0 - delta_low) * n  # eq. 11.154
                elif bl < q_full:
                    if bl <= q_int:
                        cop_int_bin = cop_low + (cop_int - cop_low) / (
                            q_int - q_low
                        ) * (
                            bl - q_low
                        )  # eq. 11.187 (2023)
                    else:  # if bl > q_int:
                        cop_int_bin = cop_int + (cop_full - cop_int) / (
                            q_full - q_int
                        ) * (
                            bl - q_int
                        )  # eq. 11.188 (2023)
                    if t <= self.heating_off_temperature or cop_int_bin < 1.0:
                        delta_int_bin = 0.0  # eq. 11.196
                    elif t > self.heating_on_temperature:
                        delta_int_bin = 1.0  # eq. 11.198
                    else:
                        delta_int_bin = 0.5  # eq. 11.197
                    rh = bl * (1.0 - delta_int_bin) * n
                    q = bl * n
                    e = q / cop_int_bin * delta_int_bin
                else:  # if bl >= q_full:
                    # TODO: allow no H4 conditions
                    # Note: builds on previously defined q_full / p_full
                    if t > fr_u(5, "°F") or t <= fr_u(17, "°F"):
                        q_full = interpolate(
                            self.net_steady_state_heating_capacity,
                            self.H4_full_cond,
                            self.H3_full_cond,
                            t,
                        )  # eq. 11.203
                        p_full = interpolate(
                            self.net_steady_state_heating_power,
                            self.H4_full_cond,
                            self.H3_full_cond,
                            t,
                        )  # eq. 11.204
                    elif t < fr_u(5, "°F"):
                        t_ratio = (t - fr_u(5.0, "°F")) / (
                            fr_u(47, "°F") - fr_u(17.0, "°F")
                        )
                        q_full = (
                            q_H4_full + (q_H1_full - q_H3_full) * t_ratio
                        )  # eq. 11.205
                        p_full = (
                            p_H4_full + (p_H1_full - p_H3_full) * t_ratio
                        )  # eq. 11.206
                    hlf_full = 1.0  # eq. 11.170
                    e = p_full * hlf_full * delta_full * n  # eq. 11.168
                    rh = (bl - q_full * hlf_full * delta_full) * n  # eq. 11.169

            q_sum += n * bl
            e_sum += e
            rh_sum += rh

        t_test = max(self.defrost.period, fr_u(90, "min"))
        t_max = min(self.defrost.max_time, fr_u(720.0, "min"))

        if self.defrost.control == DefrostControl.DEMAND:
            f_def = 1 + 0.03 * (
                1 - (t_test - fr_u(90.0, "min")) / (t_max - fr_u(90.0, "min"))
            )  # eq. 11.129
        else:
            f_def = 1  # eq. 11.130

        hspf = q_sum / (e_sum + rh_sum) * f_def  # eq. 11.133
        return to_u(hspf, "Btu/Wh")

    def print_cooling_info(self, power_units="W", capacity_units="ton_ref"):
        print(f"SEER: {self.seer():.2f}")
        for speed in range(self.number_of_cooling_speeds):
            conditions = self.make_condition(CoolingConditions, compressor_speed=speed)
            print(
                f"Net cooling power for stage {speed + 1} : {to_u(self.net_cooling_power(conditions),power_units):.2f} {power_units}"
            )
            print(
                f"Net cooling capacity for stage {speed + 1} : {to_u(self.net_total_cooling_capacity(conditions),capacity_units):.2f} {capacity_units}"
            )
            print(f"Net cooling EER for stage {speed + 1} : {self.eer(conditions):.2f}")
            print(
                f"Gross cooling COP for stage {speed + 1} : {self.gross_total_cooling_cop(conditions):.3f}"
            )
            print(f"Net SHR for stage {speed + 1} : {self.net_shr(conditions):.3f}")
        print("")

    def print_heating_info(self, power_units="W", capacity_units="ton_ref", region=4):
        print(f"HSPF (region {region}): {self.hspf(region):.2f}")
        for speed in range(self.number_of_heating_speeds):
            conditions = self.make_condition(HeatingConditions, compressor_speed=speed)
            print(
                f"Net heating power for stage {speed + 1} : {to_u(self.net_integrated_heating_power(conditions),power_units):.2f} {power_units}"
            )
            print(
                f"Net heating capacity for stage {speed + 1} : {to_u(self.net_integrated_heating_capacity(conditions),capacity_units):.2f} {capacity_units}"
            )
            print(
                f"Gross heating COP for stage {speed + 1} : {self.gross_integrated_heating_cop(conditions):.3f}"
            )
        print("")

    def generate_205_representation(self):
        timestamp = datetime.datetime.now().isoformat("T", "minutes")
        rnd = Random()
        if self.metadata.uuid_seed is None:
            self.metadata.uuid_seed = hash(self)
        rnd.seed(self.metadata.uuid_seed)
        unique_id = str(uuid.UUID(int=rnd.getrandbits(128), version=4))

        # RS0004 DX Coil

        coil_capacity = to_u(self.gross_total_cooling_capacity(), "kBtu/h")
        coil_cop = self.gross_total_cooling_cop()
        coil_shr = self.gross_shr()

        metadata_dx = {
            "data_model": "ASHRAE_205",
            "schema": "RS0004",
            "schema_version": "2.0.0",
            "description": f"{coil_capacity:.1f} kBtu/h, {coil_cop:.2f} COP, {coil_shr:.2f} SHR cooling coil",
            "id": unique_id,
            "data_timestamp": f"{timestamp}Z",
            "data_version": self.metadata.data_version,
            "data_source": self.metadata.data_source,
            "disclaimer": "This data is synthetic and does not represent any physical products.",
        }

        # if self.metadata.compressor_type is not None:
        #  description_dx = {
        #    "product_information"
        #  }

        # Create conditions
        number_of_points = 4
        outdoor_coil_entering_dry_bulb_temperatures = linspace(
            fr_u(55.0, "°F"), fr_u(125.0, "°F"), number_of_points
        ).tolist()
        indoor_coil_entering_relative_humidities = linspace(
            0.0, 1.0, number_of_points
        ).tolist()
        indoor_coil_entering_dry_bulb_temperatures = linspace(
            fr_u(65.0, "°F"), fr_u(90.0, "°F"), number_of_points
        ).tolist()
        indoor_coil_air_mass_flow_rates = linspace(
            fr_u(280.0, "cfm/ton_ref")
            * self.rated_net_total_cooling_capacity[-1]
            * STANDARD_CONDITIONS.get_rho(),
            fr_u(500.0, "cfm/ton_ref")
            * self.rated_net_total_cooling_capacity[0]
            * STANDARD_CONDITIONS.get_rho(),
            number_of_points,
        ).tolist()
        compressor_sequence_numbers = list(range(1, self.number_of_cooling_speeds + 1))
        ambient_absolute_air_pressures = linspace(
            fr_u(57226.508, "Pa"), fr_u(106868.78, "Pa"), number_of_points
        ).tolist()  # Corresponds to highest and lowest populated elevations

        grid_variables = {
            "outdoor_coil_entering_dry_bulb_temperature": outdoor_coil_entering_dry_bulb_temperatures,
            "indoor_coil_entering_relative_humidity": indoor_coil_entering_relative_humidities,
            "indoor_coil_entering_dry_bulb_temperature": indoor_coil_entering_dry_bulb_temperatures,
            "indoor_coil_air_mass_flow_rate": indoor_coil_air_mass_flow_rates,
            "compressor_sequence_number": compressor_sequence_numbers,
            "ambient_absolute_air_pressure": ambient_absolute_air_pressures,
        }

        gross_total_capacities = []
        gross_sensible_capacities = []
        gross_powers = []
        operation_states = []

        for tdb_o in outdoor_coil_entering_dry_bulb_temperatures:
            for rh_o in indoor_coil_entering_relative_humidities:
                for tdb_i in indoor_coil_entering_dry_bulb_temperatures:
                    for m_dot in indoor_coil_air_mass_flow_rates:
                        for speed in [
                            self.number_of_cooling_speeds - n
                            for n in compressor_sequence_numbers
                        ]:
                            for p in ambient_absolute_air_pressures:
                                conditions = self.make_condition(
                                    CoolingConditions,
                                    outdoor=PsychState(
                                        drybulb=tdb_o, rel_hum=0.4, pressure=p
                                    ),
                                    indoor=PsychState(
                                        drybulb=tdb_i, rel_hum=rh_o, pressure=p
                                    ),
                                    compressor_speed=speed,
                                )
                                conditions.set_mass_airflow(m_dot)

                                gross_total_capacities.append(
                                    self.gross_total_cooling_capacity(conditions)
                                )
                                gross_sensible_capacities.append(
                                    self.gross_sensible_cooling_capacity(conditions)
                                )
                                gross_powers.append(
                                    self.gross_cooling_power(conditions)
                                )
                                operation_states.append("NORMAL")

        performance_map_cooling = {
            "grid_variables": grid_variables,
            "lookup_variables": {
                "gross_total_capacity": gross_total_capacities,
                "gross_sensible_capacity": gross_sensible_capacities,
                "gross_power": gross_powers,
                "operation_state": operation_states,
            },
        }

        performance_dx = {
            "compressor_speed_control_type": (
                "DISCRETE" if self.number_of_cooling_speeds < 3 else "CONTINUOUS"
            ),  # TODO: Use staging type
            "cycling_degradation_coefficient": self.c_d_cooling,
            "performance_map_cooling": performance_map_cooling,
            "performance_map_standby": {
                "grid_variables": {
                    "outdoor_coil_environment_dry_bulb_temperature": [
                        self.crankcase_heater_setpoint_temperature + dt
                        for dt in (-0.5, 0.5)
                    ],
                },
                "lookup_variables": {
                    "gross_power": [self.crankcase_heater_capacity, 0.0],
                },
            },
        }

        representation_dx = {"metadata": metadata_dx, "performance": performance_dx}

        # RS0003 Fan Assembly
        representation_fan = self.fan.generate_205_representation()

        # RS0002 Unitary
        uuid_seed_rs0002 = hash((unique_id, representation_fan["metadata"]["id"]))
        rnd.seed(uuid_seed_rs0002)
        unique_id_rs0002 = str(uuid.UUID(int=rnd.getrandbits(128), version=4))

        metadata = {
            "data_model": "ASHRAE_205",
            "schema": "RS0002",
            "schema_version": "2.0.0",
            "description": self.metadata.description,
            "id": unique_id_rs0002,
            "data_timestamp": f"{timestamp}Z",
            "data_version": self.metadata.data_version,
            "data_source": self.metadata.data_source,
            "disclaimer": "This data is synthetic and does not represent any physical products.",
        }

        if len(self.metadata.notes) > 0:
            metadata["notes"] = self.metadata.notes

        performance = {
            "standby_power": 0.0,
            "indoor_fan_representation": representation_fan,
            "fan_position": "DRAW_THROUGH",
            "dx_system_representation": representation_dx,
        }

        representation = {"metadata": metadata, "performance": performance}

        return representation

    def plot(self, output_path: Path):
        """Generate an HTML plot for this system."""

        # Heating Temperatures
        heating_temperatures = DimensionalData(
            [
                self.heating_off_temperature,
                fr_u(5.0, "degF"),
                fr_u(17.0, "degF"),
                fr_u(35.0, "degF"),
                fr_u(47.0, "degF"),
                fr_u(60.0, "degF"),
            ],
            "Heating Temperatures",
            "K",
            "°F",
        )

        # Cooling Temperatures
        cooling_temperatures = DimensionalData(
            [
                fr_u(60.0, "degF"),
                fr_u(82.0, "degF"),
                fr_u(95.0, "degF"),
                self.cooling_off_temperature,
            ],
            "Cooling Temperatures",
            "K",
            "°F",
        )

        plot = DimensionalPlot(
            DimensionalData(
                heating_temperatures.data_values + cooling_temperatures.data_values[1:],
                "Outdoor Drybulb Temperature",
                "K",
                "°F",
            )
        )

        @dataclass
        class DisplayDataSpec:
            function: Callable[[OperatingConditions], float]
            net_or_gross: str
            version: str
            mode: str
            quantity: str

        # fmt:off
        display_specs = [
            DisplayDataSpec(self.net_steady_state_heating_capacity,"Net","Steady State","Heating","Capacity"),
            DisplayDataSpec(self.net_integrated_heating_capacity,"Net","Integrated","Heating","Capacity"),
            DisplayDataSpec(self.net_steady_state_heating_power,"Net","Steady State","Heating","Power"),
            DisplayDataSpec(self.net_integrated_heating_power,"Net","Integrated","Heating","Power"),
            DisplayDataSpec(self.net_steady_state_heating_cop,"Net","Steady State","Heating","COP"),
            DisplayDataSpec(self.net_integrated_heating_cop,"Net","Integrated","Heating","COP"),
            DisplayDataSpec(self.net_total_cooling_capacity,"Net","Total","Cooling","Capacity"),
            DisplayDataSpec(self.net_sensible_cooling_capacity,"Net","Sensible","Cooling","Capacity"),
            DisplayDataSpec(self.net_cooling_power,"Net","","Cooling","Power"),
            DisplayDataSpec(self.net_total_cooling_cop,"Net","Total","Cooling","COP"),
            DisplayDataSpec(self.net_sensible_cooling_cop,"Net","Sensible","Cooling","COP"),
            DisplayDataSpec(self.gross_steady_state_heating_capacity,"Gross","Steady State","Heating","Capacity"),
            DisplayDataSpec(self.gross_integrated_heating_capacity,"Gross","Integrated","Heating","Capacity"),
            DisplayDataSpec(self.gross_steady_state_heating_power,"Gross","Steady State","Heating","Power"),
            DisplayDataSpec(self.gross_integrated_heating_power,"Gross","Integrated","Heating","Power"),
            DisplayDataSpec(self.gross_steady_state_heating_cop,"Gross","Steady State","Heating","COP"),
            DisplayDataSpec(self.gross_integrated_heating_cop,"Gross","Integrated","Heating","COP"),
            DisplayDataSpec(self.gross_total_cooling_capacity,"Gross","Total","Cooling","Capacity"),
            DisplayDataSpec(self.gross_sensible_cooling_capacity,"Gross","Sensible","Cooling","Capacity"),
            DisplayDataSpec(self.gross_cooling_power,"Gross","","Cooling","Power"),
            DisplayDataSpec(self.gross_total_cooling_cop,"Gross","Total","Cooling","COP"),
            DisplayDataSpec(self.gross_sensible_cooling_cop,"Gross","Sensible","Cooling","COP"),
        ]
        # fmt:on

        heating_conditions = [
            [
                self.make_condition(
                    HeatingConditions,
                    compressor_speed=speed,
                    outdoor=PsychState(drybulb=tdb, rel_hum=0.4),
                )
                for tdb in heating_temperatures.data_values
            ]
            for speed in range(self.number_of_heating_speeds)
        ]
        cooling_conditions = [
            [
                self.make_condition(
                    CoolingConditions,
                    compressor_speed=speed,
                    outdoor=PsychState(drybulb=tdb, rel_hum=0.4),
                )
                for tdb in cooling_temperatures.data_values
            ]
            for speed in range(self.number_of_cooling_speeds)
        ]

        for display_spec in display_specs:
            if display_spec.mode == "Heating":
                number_of_speeds = self.number_of_heating_speeds
                conditions = heating_conditions
                pallette = "oranges"
                temperatures = heating_temperatures
                speed_names = {
                    self.heating_boost_speed: "Max",
                    self.heating_full_load_speed: "Rated",
                    self.heating_intermediate_speed: "Int.",
                    self.heating_low_speed: "Min",
                }
            elif display_spec.mode == "Cooling":
                number_of_speeds = self.number_of_cooling_speeds
                conditions = cooling_conditions
                pallette = "blues"
                temperatures = cooling_temperatures
                speed_names = {
                    self.cooling_boost_speed: "Max",
                    self.cooling_full_load_speed: "Rated",
                    self.cooling_intermediate_speed: "Int.",
                    self.cooling_low_speed: "Min",
                }
            else:
                assert False

            if display_spec.quantity == "Capacity":
                subplot_number = 1
                units = "W"
                display_units = "kBtu/h"
            elif display_spec.quantity == "Power":
                subplot_number = 2
                units = "W"
                display_units = "kW"
            elif display_spec.quantity == "COP":
                subplot_number = 3
                units = "W/W"
                display_units = "W/W"
            else:
                assert False

            is_visible = True
            if display_spec.net_or_gross == "Gross":
                is_visible = False

            line_type = "solid"
            if display_spec.version in ["Integrated", "Sensible"]:
                line_type = "dot"

            for speed in range(number_of_speeds):

                color_ratio = (number_of_speeds - speed) / number_of_speeds
                line_color = sample_colorscale(pallette, color_ratio)[0]
                if display_spec.version != "":
                    version_string = f"{display_spec.version} "
                else:
                    version_string = ""
                plot.add_display_data(
                    DisplayData(
                        [
                            display_spec.function(condition)
                            for condition in conditions[speed]
                        ],
                        name=f"{display_spec.net_or_gross} {version_string}{display_spec.mode} {display_spec.quantity} ({speed_names[speed]} Speed)",
                        native_units=units,
                        display_units=display_units,
                        x_axis=temperatures,
                        line_properties=LineProperties(
                            color=line_color, line_type=line_type, is_visible=is_visible
                        ),
                    ),
                    subplot_number=subplot_number,
                    axis_name=display_spec.quantity,
                )

        plot.write_html_plot(output_path)

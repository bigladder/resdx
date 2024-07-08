from enum import Enum
from typing import Union, Dict
from copy import deepcopy

from koozie import fr_u, to_u

from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .carrier_defrost_model import CarrierDefrostModel
from ..fan import RESNETPSCFan, RESNETBPMFan
from .tabular_data import TemperatureSpeedPerformance, make_neep_statistical_model_data
from ..enums import StagingType
from ..conditions import CoolingConditions, HeatingConditions
from ..psychrometrics import PsychState


class FanMotorType(Enum):
    PSC = 1
    BPM = 2


class RESNETDXModel(DXModel):

    def __init__(self) -> None:
        super().__init__()
        self.allowed_kwargs += [
            "rated_net_heating_capacity_17",
            "min_net_total_cooling_capacity_ratio_95",
            "min_net_total_cooling_eir_ratio_82",
            "tabular_data",
            "motor_type",
        ]
        self.net_tabular_data: Union[TemperatureSpeedPerformance, None] = None
        self.gross_tabular_data: Union[TemperatureSpeedPerformance, None] = None
        self.motor_type: Union[FanMotorType, None] = None
        self.rated_net_heating_capacity_17: Union[float, None] = None
        self.min_net_total_cooling_capacity_ratio_95: Union[float, None] = None
        self.min_net_total_cooling_eir_ratio_82: Union[float, None] = None

        self.neep_cooling_speed_map: Dict[int, float]
        self.neep_heating_speed_map: Dict[int, float]
        self.inverse_neep_cooling_speed_map: Dict[int, int]
        self.inverse_neep_heating_speed_map: Dict[int, int]

    def process_kwargs(self) -> None:
        self.net_tabular_data = self.get_kwarg_value("tabular_data")
        self.motor_type = self.get_kwarg_value("motor_type")
        self.rated_net_heating_capacity_17 = self.get_kwarg_value(
            "rated_net_heating_capacity_17"
        )
        self.min_net_total_cooling_capacity_ratio_95 = self.get_kwarg_value(
            "min_net_total_cooling_capacity_ratio_95"
        )
        self.min_net_total_cooling_eir_ratio_82 = self.get_kwarg_value(
            "min_net_total_cooling_eir_ratio_82"
        )

    # Power and capacity
    def gross_cooling_power(self, conditions: CoolingConditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.cooling_power(
                self.neep_cooling_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_cooling_correction_factor(
                conditions, NRELDXModel.gross_cooling_power
            )
        else:
            return NRELDXModel.gross_cooling_power(self, conditions)

    def gross_total_cooling_capacity(self, conditions: CoolingConditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.cooling_capacity(
                self.neep_cooling_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_cooling_correction_factor(
                conditions, NRELDXModel.gross_total_cooling_capacity
            )
        else:
            return NRELDXModel.gross_total_cooling_capacity(self, conditions)

    def gross_sensible_cooling_capacity(self, conditions):
        return NRELDXModel.gross_sensible_cooling_capacity(self, conditions)

    def gross_shr(self, conditions):
        return Title24DXModel.gross_shr(self, conditions)

    def gross_steady_state_heating_capacity(self, conditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.heating_capacity(
                self.neep_heating_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_heating_correction_factor(
                conditions, NRELDXModel.gross_steady_state_heating_capacity
            )
        else:
            return NRELDXModel.gross_steady_state_heating_capacity(self, conditions)

    def gross_integrated_heating_capacity(self, conditions: HeatingConditions):
        if self.system.defrost.in_defrost(conditions):
            fdef = max(
                min(0.134 - 0.003 * to_u(conditions.outdoor.db, "degF"), 0.08), 0.0
            )
            return self.system.gross_steady_state_heating_capacity(conditions) * (
                1.0 - 1.8 * fdef
            )
        else:
            return self.system.gross_steady_state_heating_capacity(conditions)

    def gross_steady_state_heating_power(self, conditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.heating_power(
                self.neep_heating_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_heating_correction_factor(
                conditions, NRELDXModel.gross_steady_state_heating_power
            )
        else:
            return NRELDXModel.gross_steady_state_heating_power(self, conditions)

    def gross_integrated_heating_power(self, conditions):
        if self.system.defrost.in_defrost(conditions):
            fdef = max(
                min(0.134 - 0.003 * to_u(conditions.outdoor.db, "degF"), 0.08), 0.0
            )
            return self.gross_steady_state_heating_power(conditions) * (
                1.0 - 0.3 * fdef
            )
        else:
            return self.gross_steady_state_heating_power(conditions)

    def gross_cooling_power_charge_factor(self, conditions):
        return NRELDXModel.gross_cooling_power_charge_factor(self, conditions)

    def gross_total_cooling_capacity_charge_factor(self, conditions):
        return NRELDXModel.gross_total_cooling_capacity_charge_factor(self, conditions)

    def gross_steady_state_heating_capacity_charge_factor(self, conditions):
        return NRELDXModel.gross_steady_state_heating_capacity_charge_factor(
            self, conditions
        )

    def gross_steady_state_heating_power_charge_factor(self, conditions):
        return NRELDXModel.gross_steady_state_heating_power_charge_factor(
            self, conditions
        )

    def set_net_capacities_and_fan(
        self, rated_net_total_cooling_capacity, rated_net_heating_capacity, fan
    ):
        if self.net_tabular_data is not None:
            self.system.staging_type = StagingType(
                min(self.net_tabular_data.number_of_cooling_speeds, 3)
            )
            self.system.number_of_cooling_speeds = (
                self.system.number_of_heating_speeds
            ) = (
                self.system.staging_type.value
                if self.system.staging_type != StagingType.VARIABLE_SPEED
                else 4
            )
            self.system.set_placeholder_arrays()

        if self.system.staging_type != StagingType.VARIABLE_SPEED:
            # Set high speed capacities
            self.system.rated_net_total_cooling_capacity = self.make_list(
                rated_net_total_cooling_capacity
            )

            ## Heating
            rated_net_heating_capacity = self.set_heating_default(
                rated_net_heating_capacity,
                self.system.rated_net_total_cooling_capacity[0] * 0.98
                + fr_u(180.0, "Btu/h"),
            )  # From Title24
            self.system.rated_net_heating_capacity = self.make_list(
                rated_net_heating_capacity
            )

        else:
            if not isinstance(rated_net_heating_capacity, list):
                rated_net_heating_capacity = self.set_heating_default(
                    rated_net_heating_capacity,
                    rated_net_total_cooling_capacity * 1.022 + fr_u(607.0, "Btu/h"),
                )
                if self.rated_net_heating_capacity_17 is None:
                    self.rated_net_heating_capacity_17 = (
                        0.689 * rated_net_heating_capacity
                    )
                if self.net_tabular_data is None:
                    self.net_tabular_data = make_neep_statistical_model_data(
                        rated_net_total_cooling_capacity,
                        self.system.input_seer,
                        self.system.input_eer,
                        rated_net_heating_capacity,
                        self.rated_net_heating_capacity_17,
                        self.system.input_hspf,
                        self.system.cooling_off_temperature,
                        self.system.heating_off_temperature,
                        self.min_net_total_cooling_capacity_ratio_95,
                        self.min_net_total_cooling_eir_ratio_82,
                        self.system.input_rated_net_heating_cop,
                    )

                self.system.cooling_boost_speed = 0
                self.system.cooling_full_load_speed = 1
                self.system.cooling_intermediate_speed = 2
                self.system.cooling_low_speed = 3

                self.system.heating_boost_speed = 0
                self.system.heating_full_load_speed = 1
                self.system.heating_intermediate_speed = 2
                self.system.heating_low_speed = 3

                self.neep_cooling_speed_map = self.neep_heating_speed_map = {
                    0: 3.0,
                    1: 2.0,
                    2: 1.5,
                    3: 1.0,
                }

                self.inverse_neep_cooling_speed_map = {
                    v: k
                    for k, v in self.neep_cooling_speed_map.items()
                    if v.is_integer()
                }
                self.inverse_neep_heating_speed_map = {
                    v: k
                    for k, v in self.neep_heating_speed_map.items()
                    if v.is_integer()
                }

                self.system.rated_net_total_cooling_capacity = [
                    self.net_tabular_data.cooling_capacity(
                        self.neep_cooling_speed_map[i]
                    )
                    for i in range(4)
                ]

                self.system.rated_net_heating_capacity = [
                    self.net_tabular_data.heating_capacity(
                        self.neep_heating_speed_map[i]
                    )
                    for i in range(4)
                ]

            else:
                self.system.rated_net_total_cooling_capacity = (
                    rated_net_total_cooling_capacity
                )
                self.system.rated_net_heating_capacity = rated_net_heating_capacity

        self.set_fan(fan)

        # Setup gross neep data
        if self.net_tabular_data is not None:
            self.gross_tabular_data = deepcopy(self.net_tabular_data)

            cooling_fan_powers = []
            for s_i in range(self.net_tabular_data.number_of_cooling_speeds):
                i = self.inverse_neep_cooling_speed_map[s_i + 1]
                cooling_fan_powers.append(self.system.rated_cooling_fan_power[i])
            heating_fan_powers = []
            for s_i in range(self.net_tabular_data.number_of_heating_speeds):
                i = self.inverse_neep_heating_speed_map[s_i + 1]
                heating_fan_powers.append(self.system.rated_heating_fan_power[i])

            self.gross_tabular_data.make_gross(cooling_fan_powers, heating_fan_powers)

            for i, speed in self.neep_cooling_speed_map.items():
                self.system.rated_net_cooling_cop[i] = (
                    self.net_tabular_data.cooling_cop(speed)
                )
                self.system.rated_gross_cooling_cop[i] = (
                    self.gross_tabular_data.cooling_cop(speed)
                )

            for i, speed in self.neep_heating_speed_map.items():
                self.system.rated_net_heating_cop[i] = (
                    self.net_tabular_data.heating_cop(speed)
                )
                self.system.rated_gross_heating_cop[i] = (
                    self.gross_tabular_data.heating_cop(speed)
                )

        # setup lower speed net capacities if they aren't provided
        if self.system.staging_type == StagingType.TWO_STAGE:
            if len(self.system.rated_net_total_cooling_capacity) == 1:
                self.set_lower_speed_net_capacities()

    def set_c_d_cooling(self, input):
        self.system.c_d_cooling = self.set_cooling_default(input, 0.15)

    def set_c_d_heating(self, input):
        self.system.c_d_heating = self.set_heating_default(input, 0.15)

    def set_rated_net_cooling_cop(self, input):
        if self.net_tabular_data is not None:
            pass
        else:
            NRELDXModel.set_rated_net_cooling_cop(self, input)

    def set_rated_gross_cooling_cop(self, input):
        if self.net_tabular_data is not None:
            pass
        else:
            NRELDXModel.set_rated_gross_cooling_cop(self, input)

    def set_rated_net_heating_cop(self, input):
        if self.net_tabular_data is not None:
            pass
        else:
            NRELDXModel.set_rated_net_heating_cop(self, input)

    def set_rated_gross_heating_cop(self, input):
        if self.net_tabular_data is not None:
            pass
        else:
            NRELDXModel.set_rated_gross_heating_cop(self, input)

    def set_fan(self, fan):
        # Setup fan
        self.system.rated_full_flow_external_static_pressure = (
            self.system.get_rated_full_flow_rated_pressure()
        )

        # Rated flow rates per net capacity
        self.system.rated_cooling_airflow_per_rated_net_capacity = (
            self.set_cooling_default(
                self.system.rated_cooling_airflow_per_rated_net_capacity,
                [fr_u(400.0, "cfm/ton_ref")] * self.system.number_of_cooling_speeds,
            )
        )
        self.system.rated_heating_airflow_per_rated_net_capacity = (
            self.set_heating_default(
                self.system.rated_heating_airflow_per_rated_net_capacity,
                [fr_u(400.0, "cfm/ton_ref")] * self.system.number_of_heating_speeds,
            )
        )

        if fan is not None:
            self.system.fan = fan
            for i in range(self.system.number_of_cooling_speeds):
                self.system.rated_cooling_airflow[i] = self.system.fan.airflow(
                    self.system.rated_cooling_fan_speed[i]
                )
                self.system.rated_cooling_fan_power[i] = self.system.fan.power(
                    self.system.rated_cooling_fan_speed[i]
                )
            for i in range(self.system.number_of_heating_speeds):
                self.system.rated_heating_airflow[i] = self.system.fan.airflow(
                    self.system.rated_heating_fan_speed[i]
                )
                self.system.rated_heating_fan_power[i] = self.system.fan.power(
                    self.system.rated_heating_fan_speed[i]
                )
        else:
            self.system.cooling_fan_speed = [
                None
            ] * self.system.number_of_cooling_speeds
            self.system.heating_fan_speed = [
                None
            ] * self.system.number_of_heating_speeds
            self.system.rated_cooling_fan_speed = [
                None
            ] * self.system.number_of_cooling_speeds
            self.system.rated_heating_fan_speed = [
                None
            ] * self.system.number_of_heating_speeds

            cfs = self.system.cooling_full_load_speed
            hfs = self.system.heating_full_load_speed

            self.system.rated_cooling_airflow[cfs] = (
                self.system.rated_net_total_cooling_capacity[cfs]
                * self.system.rated_cooling_airflow_per_rated_net_capacity[cfs]
            )
            self.system.rated_heating_airflow[hfs] = (
                self.system.rated_net_heating_capacity[hfs]
                * self.system.rated_heating_airflow_per_rated_net_capacity[hfs]
            )

            if (
                self.system.rated_cooling_airflow[cfs]
                >= self.system.rated_heating_airflow[hfs]
            ):
                fan_design_airflow = self.system.rated_cooling_airflow[cfs]
                rated_airflow_is_cooling = True
            else:
                fan_design_airflow = self.system.rated_heating_airflow[hfs]
                rated_airflow_is_cooling = False

            if self.motor_type is None:
                if self.system.staging_type == StagingType.SINGLE_STAGE:
                    self.motor_type = FanMotorType.PSC
                else:
                    self.motor_type = FanMotorType.BPM

            if self.motor_type == FanMotorType.PSC:
                self.system.fan = RESNETPSCFan(fan_design_airflow)
            elif self.motor_type == FanMotorType.BPM:
                self.system.fan = RESNETBPMFan(fan_design_airflow)

            if rated_airflow_is_cooling:
                self.system.cooling_fan_speed[cfs] = (
                    self.system.fan.number_of_speeds - 1
                )
                self.system.fan.add_speed(self.system.rated_heating_airflow[hfs])
                self.system.heating_fan_speed[hfs] = (
                    self.system.fan.number_of_speeds - 1
                )
            else:
                self.system.heating_fan_speed[hfs] = (
                    self.system.fan.number_of_speeds - 1
                )
                self.system.fan.add_speed(self.system.rated_cooling_airflow[cfs])
                self.system.cooling_fan_speed[cfs] = (
                    self.system.fan.number_of_speeds - 1
                )

            # At rated pressure
            self.system.rated_cooling_external_static_pressure[cfs] = (
                self.system.calculate_rated_pressure(
                    self.system.rated_cooling_airflow[cfs],
                    fan_design_airflow,
                )
            )
            self.system.fan.add_speed(
                self.system.rated_cooling_airflow[cfs],
                external_static_pressure=self.system.rated_cooling_external_static_pressure[
                    cfs
                ],
            )
            self.system.rated_cooling_fan_speed[cfs] = (
                self.system.fan.number_of_speeds - 1
            )
            self.system.rated_cooling_fan_power[cfs] = self.system.fan.power(
                self.system.rated_cooling_fan_speed[cfs],
                self.system.rated_cooling_external_static_pressure[cfs],
            )

            self.system.rated_heating_external_static_pressure[hfs] = (
                self.system.calculate_rated_pressure(
                    self.system.rated_heating_airflow[hfs],
                    fan_design_airflow,
                )
            )
            self.system.fan.add_speed(
                self.system.rated_heating_airflow[hfs],
                external_static_pressure=self.system.rated_heating_external_static_pressure[
                    hfs
                ],
            )
            self.system.rated_heating_fan_speed[hfs] = (
                self.system.fan.number_of_speeds - 1
            )
            self.system.rated_heating_fan_power[hfs] = self.system.fan.power(
                self.system.rated_heating_fan_speed[hfs],
                self.system.rated_heating_external_static_pressure[hfs],
            )

            # if net cooling capacities are provided for other speeds, add corresponding fan speeds
            for i, net_capacity in enumerate(
                self.system.rated_net_total_cooling_capacity
            ):
                if i == cfs:
                    continue
                self.system.rated_cooling_airflow[i] = (
                    net_capacity
                    * self.system.rated_cooling_airflow_per_rated_net_capacity[i]
                )
                self.system.fan.add_speed(self.system.rated_cooling_airflow[i])
                self.system.cooling_fan_speed[i] = self.system.fan.number_of_speeds - 1

                # At rated pressure
                self.system.rated_cooling_external_static_pressure[i] = (
                    self.system.calculate_rated_pressure(
                        self.system.rated_cooling_airflow[i],
                        self.system.rated_cooling_airflow[0],
                    )
                )
                self.system.fan.add_speed(
                    self.system.rated_cooling_airflow[i],
                    external_static_pressure=self.system.rated_cooling_external_static_pressure[
                        i
                    ],
                )
                self.system.rated_cooling_fan_speed[i] = (
                    self.system.fan.number_of_speeds - 1
                )
                self.system.rated_cooling_fan_power[i] = self.system.fan.power(
                    self.system.rated_cooling_fan_speed[i],
                    self.system.rated_cooling_external_static_pressure[i],
                )

            # if net heating capacities are provided for other speeds, add corresponding fan speeds
            for i, net_capacity in enumerate(self.system.rated_net_heating_capacity):
                if i == hfs:
                    continue
                self.system.rated_heating_airflow[i] = (
                    self.system.rated_net_heating_capacity[i]
                    * self.system.rated_heating_airflow_per_rated_net_capacity[i]
                )
                self.system.fan.add_speed(self.system.rated_heating_airflow[i])
                self.system.heating_fan_speed[i] = self.system.fan.number_of_speeds - 1

                # At rated pressure
                self.system.rated_heating_external_static_pressure[i] = (
                    self.system.calculate_rated_pressure(
                        self.system.rated_heating_airflow[i],
                        self.system.rated_heating_airflow[0],
                    )
                )
                self.system.fan.add_speed(
                    self.system.rated_heating_airflow[i],
                    external_static_pressure=self.system.rated_heating_external_static_pressure[
                        i
                    ],
                )
                self.system.rated_heating_fan_speed[i] = (
                    self.system.fan.number_of_speeds - 1
                )
                self.system.rated_heating_fan_power[i] = self.system.fan.power(
                    self.system.rated_heating_fan_speed[i],
                    self.system.rated_heating_external_static_pressure[i],
                )

    def set_lower_speed_net_capacities(self):
        # Cooling
        cooling_capacity_ratio = 0.72
        self.system.rated_cooling_fan_power[0] = self.system.fan.power(
            self.system.rated_cooling_fan_speed[0],
            self.system.rated_cooling_external_static_pressure[0],
        )
        self.system.rated_gross_total_cooling_capacity[0] = (
            self.system.rated_net_total_cooling_capacity[0]
            + self.system.rated_cooling_fan_power[0]
        )
        self.system.rated_gross_total_cooling_capacity[1] = (
            self.system.rated_gross_total_cooling_capacity[0] * cooling_capacity_ratio
        )

        # Solve for rated flow rate
        guess_airflow = (
            self.system.fan.design_airflow[self.system.cooling_fan_speed[0]]
            * cooling_capacity_ratio
        )
        self.system.rated_cooling_airflow[1] = self.system.fan.find_rated_fan_speed(
            self.system.rated_gross_total_cooling_capacity[1],
            self.system.rated_heating_airflow_per_rated_net_capacity[1],
            guess_airflow,
            self.system.rated_full_flow_external_static_pressure,
        )
        self.system.rated_cooling_fan_speed[1] = self.system.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.system.fan.add_speed(self.system.rated_cooling_airflow[1])
        self.system.cooling_fan_speed[1] = self.system.fan.number_of_speeds - 1

        self.system.rated_cooling_external_static_pressure[1] = (
            self.system.calculate_rated_pressure(
                self.system.rated_cooling_airflow[1],
                self.system.rated_cooling_airflow[0],
            )
        )
        self.system.rated_cooling_fan_power[1] = self.system.fan.power(
            self.system.rated_cooling_fan_speed[1],
            self.system.rated_cooling_external_static_pressure[1],
        )
        self.system.rated_net_total_cooling_capacity.append(
            self.system.rated_gross_total_cooling_capacity[1]
            - self.system.rated_cooling_fan_power[1]
        )

        # Heating
        heating_capacity_ratio = 0.72
        self.system.rated_heating_fan_power[0] = self.system.fan.power(
            self.system.rated_heating_fan_speed[0],
            self.system.rated_heating_external_static_pressure[0],
        )
        self.system.rated_gross_heating_capacity[0] = (
            self.system.rated_net_heating_capacity[0]
            - self.system.rated_heating_fan_power[0]
        )
        self.system.rated_gross_heating_capacity[1] = (
            self.system.rated_gross_heating_capacity[0] * heating_capacity_ratio
        )

        # Solve for rated flow rate
        guess_airflow = (
            self.system.fan.design_airflow[self.system.heating_fan_speed[0]]
            * heating_capacity_ratio
        )
        self.system.rated_heating_airflow[1] = self.system.fan.find_rated_fan_speed(
            self.system.rated_gross_heating_capacity[1],
            self.system.rated_heating_airflow_per_rated_net_capacity[1],
            guess_airflow,
            self.system.rated_full_flow_external_static_pressure,
        )
        self.system.rated_heating_fan_speed[1] = self.system.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.system.fan.add_speed(self.system.rated_heating_airflow[1])
        self.system.heating_fan_speed[1] = self.system.fan.number_of_speeds - 1

        self.system.rated_heating_external_static_pressure[1] = (
            self.system.calculate_rated_pressure(
                self.system.rated_heating_airflow[1],
                self.system.rated_heating_airflow[0],
            )
        )
        self.system.rated_heating_fan_power[1] = self.system.fan.power(
            self.system.rated_heating_fan_speed[1],
            self.system.rated_heating_external_static_pressure[1],
        )
        self.system.rated_net_heating_capacity.append(
            self.system.rated_gross_heating_capacity[1]
            + self.system.rated_heating_fan_power[1]
        )

    def get_cooling_correction_factor(self, conditions: CoolingConditions, function):
        rated_conditions = self.system.make_condition(
            CoolingConditions,
            compressor_speed=conditions.compressor_speed,
            outdoor=PsychState(drybulb=conditions.outdoor.db, rel_hum=0.4),
        )
        return function(self, conditions) / function(self, rated_conditions)

    def get_heating_correction_factor(self, conditions: HeatingConditions, function):
        rated_conditions = self.system.make_condition(
            HeatingConditions,
            compressor_speed=conditions.compressor_speed,
            outdoor=PsychState(drybulb=conditions.outdoor.db, rel_hum=0.4),
        )
        return function(self, conditions) / function(self, rated_conditions)

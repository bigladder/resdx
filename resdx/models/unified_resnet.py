from enum import Enum
from typing import Union, Dict
from copy import deepcopy

from koozie import fr_u, to_u

from ..dx_unit import DXUnit, AHRIVersion
from ..defrost import Defrost
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .carrier_defrost_model import CarrierDefrostModel
from ..fan import RESNETPSCFan, RESNETBPMFan
from .tabular_data import (
    TemperatureSpeedPerformance,
    make_neep_statistical_model_data,
    make_single_speed_model_data,
    make_two_speed_model_data,
    neep_cap47_from_cap95,
)
from ..util import set_default, make_list
from ..enums import StagingType
from ..conditions import CoolingConditions, HeatingConditions
from ..psychrometrics import PsychState


class FanMotorType(Enum):
    PSC = 1
    BPM = 2


class RESNETDXModel(DXUnit):

    def __init__(
        self,
        staging_type: StagingType | None = None,  # Allow default based on inputs
        rated_net_total_cooling_capacity: float = fr_u(3.0, "ton_ref"),
        input_seer: (
            float | None
        ) = None,  # SEER value input (may not match calculated SEER of the model)
        input_eer: (
            float | None
        ) = None,  # EER value input (may not match calculated EER of the model)
        rated_net_heating_capacity: float | None = None,
        rated_net_heating_capacity_17: float | None = None,
        heating_off_temperature: float | None = None,
        input_hspf: (
            float | None
        ) = None,  # HSPF value input (may not match calculated HSPF of the model)
        rated_net_heating_cop: float | None = None,
        rated_net_cooling_cop: float | None = None,
        rated_net_total_cooling_cop_82_min: float | None = None,
        min_net_total_cooling_capacity_ratio_95: float | None = None,
        motor_type: FanMotorType | None = None,
        is_ducted: bool = True,
        defrost_temperature: float = fr_u(40.0, "degF"),
        rating_standard: AHRIVersion = AHRIVersion.AHRI_210_240_2023,
        tabular_data: TemperatureSpeedPerformance | None = None,
    ) -> None:
        self.net_tabular_data: Union[TemperatureSpeedPerformance, None] = tabular_data
        self.gross_tabular_data: Union[TemperatureSpeedPerformance, None] = None
        self.motor_type: Union[FanMotorType, None] = motor_type
        self.rated_net_heating_capacity_17: Union[float, None] = (
            rated_net_heating_capacity_17
        )
        self.min_net_total_cooling_capacity_ratio_95: Union[float, None] = (
            min_net_total_cooling_capacity_ratio_95
        )
        self.rated_net_total_cooling_cop_82_min: Union[float, None] = (
            rated_net_total_cooling_cop_82_min
        )

        self.tabular_cooling_speed_map: Dict[int, float]
        self.tabular_heating_speed_map: Dict[int, float]
        self.inverse_tabular_cooling_speed_map: Dict[int, int]
        self.inverse_tabular_heating_speed_map: Dict[int, int]
        super().__init__(
            staging_type=staging_type,
            rated_net_total_cooling_capacity=rated_net_total_cooling_capacity,
            input_seer=input_seer,
            input_eer=input_eer,
            rated_net_cooling_cop=rated_net_cooling_cop,
            rated_net_heating_capacity=rated_net_heating_capacity,
            heating_off_temperature=heating_off_temperature,
            input_hspf=input_hspf,
            rated_net_heating_cop=rated_net_heating_cop,
            defrost=Defrost(high_temperature=defrost_temperature),
            is_ducted=is_ducted,
            rating_standard=rating_standard,
        )

    # Power and capacity
    def full_charge_gross_cooling_power(self, conditions: CoolingConditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.cooling_power(
                self.tabular_cooling_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_cooling_correction_factor(
                conditions, NRELDXModel.full_charge_gross_cooling_power
            )
        else:
            return NRELDXModel.full_charge_gross_cooling_power(self, conditions)

    def full_charge_gross_total_cooling_capacity(self, conditions: CoolingConditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.cooling_capacity(
                self.tabular_cooling_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_cooling_correction_factor(
                conditions, NRELDXModel.full_charge_gross_total_cooling_capacity
            )
        else:
            return NRELDXModel.full_charge_gross_total_cooling_capacity(
                self, conditions
            )

    def full_charge_gross_sensible_cooling_capacity(self, conditions):
        return NRELDXModel.full_charge_gross_sensible_cooling_capacity(self, conditions)

    def get_rated_gross_shr(self, conditions):
        return Title24DXModel.gross_shr(self, conditions)

    def full_charge_gross_steady_state_heating_capacity(self, conditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.heating_capacity(
                self.tabular_heating_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_heating_correction_factor(
                conditions, NRELDXModel.full_charge_gross_steady_state_heating_capacity
            )
        else:
            return NRELDXModel.full_charge_gross_steady_state_heating_capacity(
                self, conditions
            )

    @staticmethod
    def defrost_fraction(outdoor_drybulb_temperature: float) -> float:
        return max(
            min(0.134 - 0.003 * to_u(outdoor_drybulb_temperature, "degF"), 0.08), 0.0
        )

    @staticmethod
    def defrost_capacity_multiplier(outdoor_drybulb_temperature: float) -> float:
        return 1.0 - 1.8 * RESNETDXModel.defrost_fraction(outdoor_drybulb_temperature)

    @staticmethod
    def defrost_power_multiplier(outdoor_drybulb_temperature: float) -> float:
        return 1.0 - 0.3 * RESNETDXModel.defrost_fraction(outdoor_drybulb_temperature)

    def full_charge_gross_integrated_heating_capacity(
        self, conditions: HeatingConditions
    ) -> float:
        if self.defrost.in_defrost(conditions):
            return self.gross_steady_state_heating_capacity(
                conditions
            ) * self.defrost_capacity_multiplier(conditions.outdoor.db)

        return self.gross_steady_state_heating_capacity(conditions)

    def full_charge_gross_steady_state_heating_power(self, conditions):
        if self.net_tabular_data is not None:
            return self.gross_tabular_data.heating_power(
                self.tabular_heating_speed_map[conditions.compressor_speed],
                conditions.outdoor.db,
            ) * self.get_heating_correction_factor(
                conditions, NRELDXModel.gross_steady_state_heating_power
            )
        else:
            return NRELDXModel.gross_steady_state_heating_power(self, conditions)

    def full_charge_gross_integrated_heating_power(
        self, conditions: HeatingConditions
    ) -> float:
        if self.defrost.in_defrost(conditions):
            return self.gross_steady_state_heating_power(conditions) * (
                self.defrost_power_multiplier(conditions.outdoor.db)
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
            self.staging_type = StagingType(
                min(self.net_tabular_data.number_of_cooling_speeds, 3)
            )
            self.number_of_cooling_speeds = self.number_of_heating_speeds = (
                self.staging_type.value
                if self.staging_type != StagingType.VARIABLE_SPEED
                else 4
            )
            self.set_placeholder_arrays()

        if not isinstance(rated_net_heating_capacity, list):
            if self.staging_type != StagingType.VARIABLE_SPEED:
                self.rated_net_total_cooling_capacity = make_list(
                    rated_net_total_cooling_capacity
                )
                rated_net_heating_capacity = set_default(
                    rated_net_heating_capacity,
                    self.rated_net_total_cooling_capacity[0] * 0.98
                    + fr_u(180.0, "Btu/h"),
                    number_of_speeds=self.number_of_heating_speeds,
                )  # From Title24
                self.rated_net_heating_capacity = make_list(rated_net_heating_capacity)
                if self.staging_type == StagingType.SINGLE_STAGE:
                    if self.net_tabular_data is None:
                        self.net_tabular_data = make_single_speed_model_data(
                            cooling_capacity_95=rated_net_total_cooling_capacity,
                            seer2=self.input_seer,
                            eer2=self.input_eer,
                            heating_capacity_47=rated_net_heating_capacity,
                            heating_capacity_17=self.rated_net_heating_capacity_17,
                            hspf2=self.input_hspf,
                            heating_cop_47=self.input_rated_net_heating_cop,
                            cycling_degradation_coefficient=self.c_d_cooling,
                        )

                    self.cooling_full_load_speed = 0
                    self.heating_full_load_speed = 0

                    self.tabular_cooling_speed_map = self.tabular_heating_speed_map = {
                        0: 1.0,
                    }
                elif self.staging_type == StagingType.TWO_STAGE:
                    if self.net_tabular_data is None:
                        self.net_tabular_data = make_two_speed_model_data(
                            cooling_capacity_95=rated_net_total_cooling_capacity,
                            seer2=self.input_seer,
                            eer2=self.input_eer,
                            heating_capacity_47=rated_net_heating_capacity,
                            heating_capacity_17=self.rated_net_heating_capacity_17,
                            hspf2=self.input_hspf,
                            cooling_cop_82_min=self.rated_net_total_cooling_cop_82_min,
                            heating_cop_47=self.input_rated_net_heating_cop,
                        )

                    self.cooling_full_load_speed = 0
                    self.cooling_low_speed = 1

                    self.heating_full_load_speed = 0
                    self.heating_low_speed = 1

                    self.tabular_cooling_speed_map = self.tabular_heating_speed_map = {
                        0: 2.0,
                        1: 1.0,
                    }
            else:
                rated_net_heating_capacity = set_default(
                    rated_net_heating_capacity,
                    neep_cap47_from_cap95(rated_net_total_cooling_capacity),
                    number_of_speeds=self.number_of_heating_speeds,
                )
                if self.rated_net_heating_capacity_17 is None:
                    self.rated_net_heating_capacity_17 = (
                        0.689 * rated_net_heating_capacity
                    )  # Qm17rated from NEEP Statistics
                if self.net_tabular_data is None:
                    self.net_tabular_data = make_neep_statistical_model_data(
                        cooling_capacity_95=rated_net_total_cooling_capacity,
                        seer2=self.input_seer,
                        eer2=self.input_eer,
                        heating_capacity_47=rated_net_heating_capacity,
                        heating_capacity_17=self.rated_net_heating_capacity_17,
                        hspf2=self.input_hspf,
                        min_heating_temperature=self.heating_off_temperature,
                        cooling_capacity_ratio=self.min_net_total_cooling_capacity_ratio_95,
                        cooling_cop_82_min=self.rated_net_total_cooling_cop_82_min,
                        heating_cop_47=self.input_rated_net_heating_cop,
                    )

                self.cooling_boost_speed = 0
                self.cooling_full_load_speed = 1
                self.cooling_intermediate_speed = 2
                self.cooling_low_speed = 3

                self.heating_boost_speed = 0
                self.heating_full_load_speed = 1
                self.heating_intermediate_speed = 2
                self.heating_low_speed = 3

                self.tabular_cooling_speed_map = self.tabular_heating_speed_map = {
                    0: 3.0,
                    1: 2.0,
                    2: 1.5,  # 1.3333,
                    3: 1.0,
                }

            self.inverse_tabular_cooling_speed_map = {
                v: k
                for k, v in self.tabular_cooling_speed_map.items()
                if v.is_integer()
            }
            self.inverse_tabular_heating_speed_map = {
                v: k
                for k, v in self.tabular_heating_speed_map.items()
                if v.is_integer()
            }

            self.rated_net_total_cooling_capacity = [
                self.net_tabular_data.cooling_capacity(
                    self.tabular_cooling_speed_map[i]
                )
                for i in range(len(self.tabular_cooling_speed_map))
            ]

            self.rated_net_heating_capacity = [
                self.net_tabular_data.heating_capacity(
                    self.tabular_heating_speed_map[i]
                )
                for i in range(len(self.tabular_heating_speed_map))
            ]
        else:
            self.rated_net_total_cooling_capacity = rated_net_total_cooling_capacity
            self.rated_net_heating_capacity = rated_net_heating_capacity

        self.set_fan(fan)

        # Setup gross tabular data
        if self.net_tabular_data is not None:
            self.gross_tabular_data = deepcopy(self.net_tabular_data)

            cooling_fan_powers = []
            for s_i in range(self.net_tabular_data.number_of_cooling_speeds):
                i = self.inverse_tabular_cooling_speed_map[s_i + 1]
                cooling_fan_powers.append(self.rated_cooling_fan_power[i])
            heating_fan_powers = []
            for s_i in range(self.net_tabular_data.number_of_heating_speeds):
                i = self.inverse_tabular_heating_speed_map[s_i + 1]
                heating_fan_powers.append(self.rated_heating_fan_power[i])

            self.gross_tabular_data.make_gross(cooling_fan_powers, heating_fan_powers)

            for i, speed in self.tabular_cooling_speed_map.items():
                self.rated_net_cooling_cop[i] = self.net_tabular_data.cooling_cop(speed)
                self.rated_gross_cooling_cop[i] = self.gross_tabular_data.cooling_cop(
                    speed
                )

            for i, speed in self.tabular_heating_speed_map.items():
                self.rated_net_heating_cop[i] = self.net_tabular_data.heating_cop(speed)
                self.rated_gross_heating_cop[i] = self.gross_tabular_data.heating_cop(
                    speed
                )

        # setup lower speed net capacities if they aren't provided
        if self.staging_type == StagingType.TWO_STAGE:
            if len(self.rated_net_total_cooling_capacity) == 1:
                self.set_lower_speed_net_capacities()

    def set_c_d_cooling(self, c_d_cooling):
        if self.staging_type == StagingType.VARIABLE_SPEED:
            self.c_d_cooling = set_default(c_d_cooling, 0.25)
        else:
            self.c_d_cooling = set_default(c_d_cooling, 0.08)

    def set_c_d_heating(self, c_d_heating):
        if self.staging_type == StagingType.VARIABLE_SPEED:
            self.c_d_heating = set_default(c_d_heating, 0.25)
        else:
            self.c_d_heating = set_default(c_d_heating, 0.08)

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
        self.rated_full_flow_external_static_pressure = (
            self.get_rated_full_flow_rated_pressure()
        )

        # Rated flow rates per net capacity
        self.rated_cooling_airflow_per_rated_net_capacity = set_default(
            self.rated_cooling_airflow_per_rated_net_capacity,
            [fr_u(400.0, "cfm/ton_ref")] * self.number_of_cooling_speeds,
            number_of_speeds=self.number_of_cooling_speeds,
        )
        self.rated_heating_airflow_per_rated_net_capacity = set_default(
            self.rated_heating_airflow_per_rated_net_capacity,
            [fr_u(400.0, "cfm/ton_ref")] * self.number_of_heating_speeds,
            number_of_speeds=self.number_of_heating_speeds,
        )

        if fan is not None:
            self.fan = fan
            for i in range(self.number_of_cooling_speeds):
                self.rated_cooling_airflow[i] = self.fan.airflow(
                    self.rated_cooling_fan_speed[i]
                )
                self.rated_cooling_fan_power[i] = self.fan.power(
                    self.rated_cooling_fan_speed[i]
                )
            for i in range(self.number_of_heating_speeds):
                self.rated_heating_airflow[i] = self.fan.airflow(
                    self.rated_heating_fan_speed[i]
                )
                self.rated_heating_fan_power[i] = self.fan.power(
                    self.rated_heating_fan_speed[i]
                )
        else:
            self.cooling_fan_speed = [None] * self.number_of_cooling_speeds
            self.heating_fan_speed = [None] * self.number_of_heating_speeds
            self.rated_cooling_fan_speed = [None] * self.number_of_cooling_speeds
            self.rated_heating_fan_speed = [None] * self.number_of_heating_speeds

            cfs = self.cooling_full_load_speed
            hfs = self.heating_full_load_speed

            self.rated_cooling_airflow[cfs] = (
                self.rated_net_total_cooling_capacity[cfs]
                * self.rated_cooling_airflow_per_rated_net_capacity[cfs]
            )
            self.rated_heating_airflow[hfs] = (
                self.rated_net_heating_capacity[hfs]
                * self.rated_heating_airflow_per_rated_net_capacity[hfs]
            )

            if self.rated_cooling_airflow[cfs] >= self.rated_heating_airflow[hfs]:
                fan_design_airflow = self.rated_cooling_airflow[cfs]
                rated_airflow_is_cooling = True
            else:
                fan_design_airflow = self.rated_heating_airflow[hfs]
                rated_airflow_is_cooling = False

            if self.motor_type is None:
                if self.staging_type == StagingType.SINGLE_STAGE:
                    self.motor_type = FanMotorType.PSC
                else:
                    self.motor_type = FanMotorType.BPM

            if self.motor_type == FanMotorType.PSC:
                self.fan = RESNETPSCFan(fan_design_airflow)
            elif self.motor_type == FanMotorType.BPM:
                self.fan = RESNETBPMFan(fan_design_airflow)

            if rated_airflow_is_cooling:
                self.cooling_fan_speed[cfs] = self.fan.number_of_speeds - 1
                self.fan.add_speed(self.rated_heating_airflow[hfs])
                self.heating_fan_speed[hfs] = self.fan.number_of_speeds - 1
            else:
                self.heating_fan_speed[hfs] = self.fan.number_of_speeds - 1
                self.fan.add_speed(self.rated_cooling_airflow[cfs])
                self.cooling_fan_speed[cfs] = self.fan.number_of_speeds - 1

            # At rated pressure
            self.rated_cooling_external_static_pressure[cfs] = (
                self.rated_full_flow_external_static_pressure
            )
            self.fan.add_speed(
                self.rated_cooling_airflow[cfs],
                external_static_pressure=self.rated_cooling_external_static_pressure[
                    cfs
                ],
            )
            self.rated_cooling_fan_speed[cfs] = self.fan.number_of_speeds - 1
            self.rated_cooling_fan_power[cfs] = self.fan.power(
                self.rated_cooling_fan_speed[cfs],
                self.rated_cooling_external_static_pressure[cfs],
            )

            self.rated_heating_external_static_pressure[hfs] = (
                self.calculate_rated_pressure(
                    self.rated_heating_airflow[hfs],
                    self.rated_cooling_airflow[cfs],
                )
            )
            self.fan.add_speed(
                self.rated_heating_airflow[hfs],
                external_static_pressure=self.rated_heating_external_static_pressure[
                    hfs
                ],
            )
            self.rated_heating_fan_speed[hfs] = self.fan.number_of_speeds - 1
            self.rated_heating_fan_power[hfs] = self.fan.power(
                self.rated_heating_fan_speed[hfs],
                self.rated_heating_external_static_pressure[hfs],
            )

            # if net cooling capacities are provided for other speeds, add corresponding fan speeds
            for i, net_capacity in enumerate(self.rated_net_total_cooling_capacity):
                if i == cfs:
                    continue
                self.rated_cooling_airflow[i] = (
                    net_capacity * self.rated_cooling_airflow_per_rated_net_capacity[i]
                )
                self.fan.add_speed(self.rated_cooling_airflow[i])
                self.cooling_fan_speed[i] = self.fan.number_of_speeds - 1

                # At rated pressure
                self.rated_cooling_external_static_pressure[i] = (
                    self.calculate_rated_pressure(
                        self.rated_cooling_airflow[i],
                        self.rated_cooling_airflow[cfs],
                    )
                )
                self.fan.add_speed(
                    self.rated_cooling_airflow[i],
                    external_static_pressure=self.rated_cooling_external_static_pressure[
                        i
                    ],
                )
                self.rated_cooling_fan_speed[i] = self.fan.number_of_speeds - 1
                self.rated_cooling_fan_power[i] = self.fan.power(
                    self.rated_cooling_fan_speed[i],
                    self.rated_cooling_external_static_pressure[i],
                )

            # if net heating capacities are provided for other speeds, add corresponding fan speeds
            for i, net_capacity in enumerate(self.rated_net_heating_capacity):
                if i == hfs:
                    continue
                self.rated_heating_airflow[i] = (
                    self.rated_net_heating_capacity[i]
                    * self.rated_heating_airflow_per_rated_net_capacity[i]
                )
                self.fan.add_speed(self.rated_heating_airflow[i])
                self.heating_fan_speed[i] = self.fan.number_of_speeds - 1

                # At rated pressure
                self.rated_heating_external_static_pressure[i] = (
                    self.calculate_rated_pressure(
                        self.rated_heating_airflow[i],
                        self.rated_cooling_airflow[cfs],
                    )
                )
                self.fan.add_speed(
                    self.rated_heating_airflow[i],
                    external_static_pressure=self.rated_heating_external_static_pressure[
                        i
                    ],
                )
                self.rated_heating_fan_speed[i] = self.fan.number_of_speeds - 1
                self.rated_heating_fan_power[i] = self.fan.power(
                    self.rated_heating_fan_speed[i],
                    self.rated_heating_external_static_pressure[i],
                )

    def set_lower_speed_net_capacities(self):
        # Cooling
        cooling_capacity_ratio = 0.72
        self.rated_cooling_fan_power[0] = self.fan.power(
            self.rated_cooling_fan_speed[0],
            self.rated_cooling_external_static_pressure[0],
        )
        self.rated_gross_total_cooling_capacity[0] = (
            self.rated_net_total_cooling_capacity[0] + self.rated_cooling_fan_power[0]
        )
        self.rated_gross_total_cooling_capacity[1] = (
            self.rated_gross_total_cooling_capacity[0] * cooling_capacity_ratio
        )

        # Solve for rated flow rate
        guess_airflow = (
            self.fan.design_airflow[self.cooling_fan_speed[0]] * cooling_capacity_ratio
        )
        self.rated_cooling_airflow[1] = self.fan.find_rated_fan_speed(
            self.rated_gross_total_cooling_capacity[1],
            self.rated_heating_airflow_per_rated_net_capacity[1],
            guess_airflow,
            self.rated_full_flow_external_static_pressure,
        )
        self.rated_cooling_fan_speed[1] = self.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.fan.add_speed(self.rated_cooling_airflow[1])
        self.cooling_fan_speed[1] = self.fan.number_of_speeds - 1

        self.rated_cooling_external_static_pressure[1] = self.calculate_rated_pressure(
            self.rated_cooling_airflow[1],
            self.rated_cooling_airflow[0],
        )
        self.rated_cooling_fan_power[1] = self.fan.power(
            self.rated_cooling_fan_speed[1],
            self.rated_cooling_external_static_pressure[1],
        )
        self.rated_net_total_cooling_capacity.append(
            self.rated_gross_total_cooling_capacity[1] - self.rated_cooling_fan_power[1]
        )

        # Heating
        heating_capacity_ratio = 0.72
        self.rated_heating_fan_power[0] = self.fan.power(
            self.rated_heating_fan_speed[0],
            self.rated_heating_external_static_pressure[0],
        )
        self.rated_gross_heating_capacity[0] = (
            self.rated_net_heating_capacity[0] - self.rated_heating_fan_power[0]
        )
        self.rated_gross_heating_capacity[1] = (
            self.rated_gross_heating_capacity[0] * heating_capacity_ratio
        )

        # Solve for rated flow rate
        guess_airflow = (
            self.fan.design_airflow[self.heating_fan_speed[0]] * heating_capacity_ratio
        )
        self.rated_heating_airflow[1] = self.fan.find_rated_fan_speed(
            self.rated_gross_heating_capacity[1],
            self.rated_heating_airflow_per_rated_net_capacity[1],
            guess_airflow,
            self.rated_full_flow_external_static_pressure,
        )
        self.rated_heating_fan_speed[1] = self.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.fan.add_speed(self.rated_heating_airflow[1])
        self.heating_fan_speed[1] = self.fan.number_of_speeds - 1

        self.rated_heating_external_static_pressure[1] = self.calculate_rated_pressure(
            self.rated_heating_airflow[1],
            self.rated_cooling_airflow[0],
        )
        self.rated_heating_fan_power[1] = self.fan.power(
            self.rated_heating_fan_speed[1],
            self.rated_heating_external_static_pressure[1],
        )
        self.rated_net_heating_capacity.append(
            self.rated_gross_heating_capacity[1] + self.rated_heating_fan_power[1]
        )

    def get_cooling_correction_factor(self, conditions: CoolingConditions, function):
        rated_conditions = self.make_condition(
            CoolingConditions,
            compressor_speed=conditions.compressor_speed,
            outdoor=PsychState(drybulb=conditions.outdoor.db, rel_hum=0.4),
        )
        return function(self, conditions) / function(self, rated_conditions)

    def get_heating_correction_factor(self, conditions: HeatingConditions, function):
        rated_conditions = self.make_condition(
            HeatingConditions,
            compressor_speed=conditions.compressor_speed,
            outdoor=PsychState(drybulb=conditions.outdoor.db, rel_hum=0.4),
        )
        return function(self, conditions) / function(self, rated_conditions)

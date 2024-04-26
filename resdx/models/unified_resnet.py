from enum import Enum
from typing import Union

from koozie import fr_u

from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .carrier_defrost_model import CarrierDefrostModel
from ..fan import RESNETPSCFan, RESNETBPMFan
from .neep_data import NEEPPerformance, make_neep_statistical_model_data
from ..enums import StagingType


class FanMotorType(Enum):
    PSC = 1
    BPM = 2


class RESNETDXModel(DXModel):

    def __init__(self) -> None:
        super().__init__()
        self.allowed_kwargs += [
            "neep_data",
            "rated_net_heating_capacity_17",
            "rated_net_heating_cop_47" "min_net_total_cooling_capacity_ratio_95",
            "min_net_total_cooling_eir_ratio_82" "neep_data",
            "motor_type",
        ]
        self.neep_data: Union[NEEPPerformance, None] = None
        self.motor_type: Union[FanMotorType, None] = None
        self.rated_net_heating_capacity_17: Union[float, None] = None
        self.rated_net_heating_cop_47: Union[float, None] = None
        self.min_net_total_cooling_capacity_ratio_95: Union[float, None] = None
        self.min_net_total_cooling_eir_ratio_82: Union[float, None] = None

    def process_kwargs(self) -> None:
        self.neep_data = self.get_kwarg_value("neep_data")
        self.motor_type = self.get_kwarg_value("motor_type")
        self.rated_net_heating_capacity_17 = self.get_kwarg_value(
            "rated_net_heating_capacity_17"
        )
        self.rated_net_heating_cop_47 = self.get_kwarg_value("rated_net_heating_cop_47")
        self.min_net_total_cooling_capacity_ratio_95 = self.get_kwarg_value(
            "min_net_total_cooling_capacity_ratio_95"
        )
        self.min_net_total_cooling_eir_ratio_82 = self.get_kwarg_value(
            "min_net_total_cooling_eir_ratio_82"
        )

    # Power and capacity
    def gross_cooling_power(self, conditions):
        return NRELDXModel.gross_cooling_power(self, conditions)

    def gross_total_cooling_capacity(self, conditions):
        return NRELDXModel.gross_total_cooling_capacity(self, conditions)

    def gross_sensible_cooling_capacity(self, conditions):
        return NRELDXModel.gross_sensible_cooling_capacity(self, conditions)

    def gross_shr(self, conditions):
        return Title24DXModel.gross_shr(self, conditions)

    def gross_steady_state_heating_capacity(self, conditions):
        return NRELDXModel.gross_steady_state_heating_capacity(self, conditions)

    def gross_integrated_heating_capacity(self, conditions):
        return CarrierDefrostModel.gross_integrated_heating_capacity(self, conditions)

    def gross_steady_state_heating_power(self, conditions):
        return NRELDXModel.gross_steady_state_heating_power(self, conditions)

    def gross_integrated_heating_power(self, conditions):
        return CarrierDefrostModel.gross_integrated_heating_power(self, conditions)

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
        # Set high speed capacities

        ## Cooling (high speed net total cooling capacity is required)
        if type(rated_net_total_cooling_capacity) is list:
            self.system.rated_net_total_cooling_capacity = (
                rated_net_total_cooling_capacity
            )
        else:
            # Even if the system has more than one speed, the first speed will be the input
            self.system.rated_net_total_cooling_capacity = [
                rated_net_total_cooling_capacity
            ]

        ## Heating
        rated_net_heating_capacity = self.set_heating_default(
            rated_net_heating_capacity,
            self.system.rated_net_total_cooling_capacity[0] * 0.98
            + fr_u(180.0, "Btu/h"),
        )  # From Title24
        if type(rated_net_heating_capacity) is list:
            self.system.rated_net_heating_capacity = rated_net_heating_capacity
        else:
            # Even if the system has more than one speed, the first speed will be the input
            self.system.rated_net_heating_capacity = [rated_net_heating_capacity]

        # setup fan
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

            self.system.rated_cooling_airflow[0] = (
                self.system.rated_net_total_cooling_capacity[0]
                * self.system.rated_cooling_airflow_per_rated_net_capacity[0]
            )
            if self.motor_type is None:
                if self.system.number_of_cooling_speeds > 1:
                    self.motor_type = FanMotorType.BPM
                else:
                    self.motor_type = FanMotorType.PSC

            if self.motor_type == FanMotorType.PSC:
                self.system.fan = RESNETPSCFan(self.system.rated_cooling_airflow[0])
            elif self.motor_type == FanMotorType.BPM:
                self.system.fan = RESNETBPMFan(self.system.rated_cooling_airflow[0])

            self.system.cooling_fan_speed[0] = self.system.fan.number_of_speeds - 1

            self.system.rated_heating_airflow[0] = (
                self.system.rated_net_heating_capacity[0]
                * self.system.rated_heating_airflow_per_rated_net_capacity[0]
            )
            self.system.fan.add_speed(self.system.rated_heating_airflow[0])
            self.system.heating_fan_speed[0] = self.system.fan.number_of_speeds - 1

            # At rated pressure
            self.system.rated_cooling_external_static_pressure[0] = (
                self.system.calculate_rated_pressure(
                    self.system.rated_cooling_airflow[0],
                    self.system.rated_cooling_airflow[0],
                )
            )
            self.system.fan.add_speed(
                self.system.rated_cooling_airflow[0],
                external_static_pressure=self.system.rated_cooling_external_static_pressure[
                    0
                ],
            )
            self.system.rated_cooling_fan_speed[0] = (
                self.system.fan.number_of_speeds - 1
            )
            self.system.rated_cooling_fan_power[0] = self.system.fan.power(
                self.system.rated_cooling_fan_speed[0],
                self.system.rated_cooling_external_static_pressure[0],
            )

            self.system.rated_heating_external_static_pressure[0] = (
                self.system.calculate_rated_pressure(
                    self.system.rated_heating_airflow[0],
                    self.system.rated_heating_airflow[0],
                )
            )
            self.system.fan.add_speed(
                self.system.rated_heating_airflow[0],
                external_static_pressure=self.system.rated_heating_external_static_pressure[
                    0
                ],
            )
            self.system.rated_heating_fan_speed[0] = (
                self.system.fan.number_of_speeds - 1
            )
            self.system.rated_heating_fan_power[0] = self.system.fan.power(
                self.system.rated_heating_fan_speed[0],
                self.system.rated_heating_external_static_pressure[0],
            )

            # if net cooling capacities are provided for other speeds, add corresponding fan speeds
            for i, net_capacity in enumerate(
                self.system.rated_net_total_cooling_capacity[1:]
            ):
                i += 1  # Since we're starting at the second item
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
            for i, net_capacity in enumerate(
                self.system.rated_net_heating_capacity[1:]
            ):
                i += 1  # Since we're starting at the second item
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

        # setup lower speed net capacities if they aren't provided
        if (
            len(self.system.rated_net_total_cooling_capacity)
            < self.system.number_of_cooling_speeds
        ):
            if self.system.staging_type == StagingType.TWO_STAGE:
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
                    self.system.rated_gross_total_cooling_capacity[0]
                    * cooling_capacity_ratio
                )

                # Solve for rated flow rate
                guess_airflow = (
                    self.system.fan.design_airflow[self.system.cooling_fan_speed[0]]
                    * cooling_capacity_ratio
                )
                self.system.rated_cooling_airflow[1] = (
                    self.system.fan.find_rated_fan_speed(
                        self.system.rated_gross_total_cooling_capacity[1],
                        self.system.rated_heating_airflow_per_rated_net_capacity[1],
                        guess_airflow,
                        self.system.rated_full_flow_external_static_pressure,
                    )
                )
                self.system.rated_cooling_fan_speed[1] = (
                    self.system.fan.number_of_speeds - 1
                )

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
                self.system.rated_heating_airflow[1] = (
                    self.system.fan.find_rated_fan_speed(
                        self.system.rated_gross_heating_capacity[1],
                        self.system.rated_heating_airflow_per_rated_net_capacity[1],
                        guess_airflow,
                        self.system.rated_full_flow_external_static_pressure,
                    )
                )
                self.system.rated_heating_fan_speed[1] = (
                    self.system.fan.number_of_speeds - 1
                )

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
            else:
                if self.neep_data is None:
                    self.neep_data = make_neep_statistical_model_data(
                        self.system.rated_net_total_cooling_capacity[0],
                        self.system.input_seer,
                        self.system.input_eer,
                        self.system.rated_net_heating_capacity[0],
                        self.rated_net_heating_capacity_17,
                        self.system.input_hspf,
                        self.system.cooling_off_temperature,
                        self.system.heating_off_temperature,
                        self.min_net_total_cooling_capacity_ratio_95,
                        self.min_net_total_cooling_eir_ratio_82,
                        self.rated_net_heating_cop_47,
                    )


    def set_c_d_cooling(self, input):
        self.system.c_d_cooling = self.set_cooling_default(input, 0.15)

    def set_c_d_heating(self, input):
        self.system.c_d_heating = self.set_heating_default(input, 0.15)

    def set_rated_net_cooling_cop(self, input):
        NRELDXModel.set_rated_net_cooling_cop(self, input)

    def set_rated_gross_cooling_cop(self, input):
        NRELDXModel.set_rated_gross_cooling_cop(self, input)

    def set_rated_net_heating_cop(self, input):
        NRELDXModel.set_rated_net_heating_cop(self, input)

    def set_rated_gross_heating_cop(self, input):
        NRELDXModel.set_rated_gross_heating_cop(self, input)

from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .carrier_defrost_model import CarrierDefrostModel
from koozie import fr_u
from ..fan import ECMFlowFan, PSCFan


class RESNETDXModel(DXModel):

    def __init__(self):
        super().__init__()
        self.allowed_kwargs += [
            "eer2",
            "seer2",
            "hspf2",
            "rated_net_stead_state_heating_capacity_17",
            "rated_net_total_cooling_min_capacity_ratio_95",
        ]

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
            design_external_static_pressure = fr_u(0.5, "in_H2O")
            if self.system.number_of_cooling_speeds > 1:
                self.system.fan = ECMFlowFan(
                    self.system.rated_cooling_airflow[0],
                    design_external_static_pressure,
                    design_efficacy=fr_u(0.3, "W/cfm"),
                )
            else:
                self.system.fan = PSCFan(
                    self.system.rated_cooling_airflow[0],
                    design_external_static_pressure,
                    design_efficacy=fr_u(0.365, "W/cfm"),
                )

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

            # if net cooling capacities are provided for other speeds, add corresponding fan speeds
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
            # Cooling
            if self.system.number_of_cooling_speeds == 2:
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
            if self.system.number_of_heating_speeds == 2:
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
                raise Exception(
                    f"No default rated net total cooling capacities for systems with more than two speeds"
                )

    def set_variable_capacity_cooling_properties(self):
        cooling_temperatures = [
            fr_u(60, "degF"),
            fr_u(82, "degF"),
            fr_u(95, "degF"),
            self.system.cooling_off_temperature,
        ]

        Tmin = 0
        T82 = 1
        T95 = 2
        Tmax = 3

        Qmin = 0
        Qrated = 1
        Qmax = 2

        if "rated_net_total_cooling_min_capacity_ratio_95" in self.system.kwargs:
            Qr95min = self.system.kwargs[
                "rated_net_total_cooling_min_capacity_ratio_95"
            ]
        else:
            Qr95min = (
                0.510 - 0.119 * self.system.kwargs["seer2"] / self.system.kwargs["eer2"]
            )

        if "rated_net_total_cooling_min_eir_ratio_82" in self.system.kwargs:
            EIRr82min = self.system.kwargs[
                "rated_net_total_cooling_min_eir_ratio_82"
            ]
        else:
            EIRr82min = (
                1.305 - 0.324 * self.system.kwargs["seer2"] / self.system.kwargs["eer2"]
            )


        Qr95rated = 0.934
        Qm95max = 0.940
        Qm95min = 0.948
        EIRr95rated = 0.928
        EIRm95max = 1.326
        EIRm95min = 1.315

        Q = [[None] * (Qmax + 1) for _ in range(Tmax + 1)]  # Initialize dimensions
        P = Q

        # Net Total Capacity

        # 95 F
        Q[T95][Qrated] = self.system.rated_net_total_cooling_capacity
        Q[T95][Qmax] = Q[T95][Qrated] / Qr95rated
        Q[T95][Qmin] = Q[T95][Qmax] * Qr95min

        # 82 F
        Q[T82][Qmax] = Q[T95][Qmax] / Qm95max
        Q[T82][Qmin] = Q[T95][Qmin] / Qm95min
        Q[T82][Qrated] = Q[T82][Qmin] + (Q[T95][Qrated] - Q[T95][Qmin]) / (
            Q[T95][Qmax] - Q[T95][Qmin]
        ) * (
            Q[T82][Qmax] - Q[T82][Qmin]
        )  # Interpolate

        # Tmin
        Q[Tmin][Qmax] = extrapolate_below(Q, Qmax, cooling_temperatures)
        Q[Tmin][Qrated] = extrapolate_below(Q, Qrated, cooling_temperatures)
        Q[Tmin][Qmin] = extrapolate_below(Q, Qmin, cooling_temperatures)

        # Tmax
        Q[Tmax][Qmax] = extrapolate_above(Q, Qmax, cooling_temperatures)
        Q[Tmax][Qrated] = extrapolate_above(Q, Qrated, cooling_temperatures)
        Q[Tmax][Qmin] = extrapolate_above(Q, Qmin, cooling_temperatures)

        # Net Power
        Pr95rated = Qr95rated * EIRr95rated
        Pr82min = (Q[T82][Qmin]/Q[T82][Qmax]) * EIRr82min
        Pm95min = Qm95min * EIRm95min
        Pm95max = Qm95max * EIRm95max

        # 95/82 F
        P[T95][Qrated] = Q[T95][Qrated] / fr_u(self.system.kwargs["eer2"], "Wh/Btu")
        P[T95][Qmax] = P[T95][Qrated] / Pr95rated
        P[T82][Qmax] = P[T95][Qmax] / Pm95max
        P[T82][Qmin] = P[T82][Qmax] * Pr82min
        P[T95][Qmin] = P[T82][Qmin] * Pm95min

        P[T82][Qrated] = P[T82][Qmin] + (P[T95][Qrated] - P[T95][Qmin]) / (
            P[T95][Qmax] - P[T95][Qmin]
        ) * (
            P[T82][Qmax] - P[T82][Qmin]
        )  # Interpolate

        # Tmin
        P[Tmin][Qmax] = extrapolate_below(P, Qmax, cooling_temperatures)
        P[Tmin][Qrated] = extrapolate_below(P, Qrated, cooling_temperatures)
        P[Tmin][Qmin] = extrapolate_below(P, Qmin, cooling_temperatures)

        # Tmax
        P[Tmax][Qmax] = extrapolate_above(P, Qmax, cooling_temperatures)
        P[Tmax][Qrated] = extrapolate_above(P, Qrated, cooling_temperatures)
        P[Tmax][Qmin] = extrapolate_above(P, Qmin, cooling_temperatures)

    def set_variable_capacity_heating_properties(self):
        heating_temperatures = [
            self.system.heating_off_temperature,
            fr_u(5, "degF"),
            fr_u(17, "degF"),
            fr_u(47, "degF"),
            fr_u(60, "degF"),
        ]

        Tmin = 0
        T5 = 1
        T17 = 2
        T47 = 3
        Tmax = 4

        Qmin = 0
        Qrated = 1
        Qmax = 2

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

        if "rated_net_stead_state_heating_capacity_17" in self.system.kwargs:
            Qm17rated = (
                self.system.kwargs["rated_net_stead_state_heating_capacity_17"]
                / self.system.rated_net_total_cooling_capacity
            )
        else:
            Qm17rated = 0.689

        Q = [[None] * (Qmax + 1) for _ in range(Tmax + 1)]  # Initialize dimensions
        P = Q

        # Net Total Capacity

        # 47 F
        Q[T47][Qrated] = self.system.rated_net_total_cooling_capacity
        Q[T47][Qmax] = Q[T47][Qrated] / Qr47rated
        Q[T47][Qmin] = Q[T47][Qmax] * Qr47min

        # 17 F
        Q[T17][Qrated] = Q[T47][Qrated] * Qm17rated
        Q[T17][Qmax] = Q[T17][Qrated] / Qr17rated
        Q[T17][Qmin] = Q[T17][Qmax] * Qr17min

        # 5 F
        Q[T5][Qmax] = Q[T17][Qmax] * Qm5max
        Q[T5][Qrated] = Q[T5][Qmax] * Qr5rated
        Q[T5][Qmin] = Q[T5][Qmax] * Qr5min

        QmLCTmax = 1 / (
            1
            - fr_u(QmslopeLCTmax, "1/degF")
            * (heating_temperatures[T5] - heating_temperatures[Tmin])
        )
        QmLCTmin = 1 / (
            1
            - fr_u(QmslopeLCTmin, "1/degF")
            * (heating_temperatures[T5] - heating_temperatures[Tmin])
        )

        # Tmin
        Q[Tmin][Qmax] = Q[T5][Qmax] * QmLCTmax
        Q[Tmin][Qmin] = Q[T5][Qmin] * QmLCTmin
        Q[Tmin][Qrated] = Q[Tmin][Qmin] + (Q[T5][Qrated] - Q[T5][Qmin]) / (
            Q[T5][Qmax] - Q[T5][Qmin]
        ) * (
            Q[Tmin][Qmax] - Q[Tmin][Qmin]
        )  # Interpolate

        # Tmax
        Q[Tmax][Qmax] = Q[T47][Qmax]
        Q[Tmax][Qrated] = Q[T47][Qrated]
        Q[Tmax][Qmin] = Q[T47][Qmin]

        # Net Power
        if self.system.rated_net_heating_cop is None:
            self.system.rated_net_heating_cop = (
                2.837 + 0.066 * self.system.kwargs["hspf2"]
            )

        Pr47rated = Qr47rated * EIRr47rated
        Pr47min = Qr47min * EIRr47min
        Pm17rated = Qm17rated * EIRm17rated
        Pr17rated = Qr17rated * EIRr17rated
        Pr17min = Qr17min * EIRr17min
        Pm5max = Qm5max * EIRm5max
        Pr5rated = Qr5rated * EIRr5rated
        Pr5min = Qr5min * EIRr5min

        # 47 F
        P[T47][Qrated] = Q[T47][Qrated] / self.system.rated_net_heating_cop
        P[T47][Qmax] = P[T47][Qrated] / Pr47rated
        P[T47][Qmin] = P[T47][Qmax] * Pr47min

        # 17 F
        P[T17][Qrated] = P[T47][Qrated] * Pm17rated
        P[T17][Qmax] = P[T17][Qrated] / Pr17rated
        P[T17][Qmin] = P[T17][Qmax] * Pr17min

        # 5 F
        P[T5][Qmax] = P[T17][Qmax] * Pm5max
        P[T5][Qrated] = P[T5][Qmax] * Pr5rated
        P[T5][Qmin] = P[T5][Qmax] * Pr5min

        EIRmLCTmax = 1 / (
            1
            - fr_u(EIRmslopeLCTmax, "1/degF")
            * (heating_temperatures[T5] - heating_temperatures[Tmin])
        )
        EIRmLCTmin = 1 / (
            1
            - fr_u(EIRmslopeLCTmin, "1/degF")
            * (heating_temperatures[T5] - heating_temperatures[Tmin])
        )

        PmLCTmax = EIRmLCTmax * (Q[Tmin][Qmax] / Q[T5][Qmax])
        PmLCTmin = EIRmLCTmin * (Q[Tmin][Qmin] / Q[T5][Qmin])

        # Tmin
        P[Tmin][Qmax] = P[T5][Qmax] * PmLCTmax
        P[Tmin][Qmin] = P[T5][Qmin] * PmLCTmin
        P[Tmin][Qrated] = P[Tmin][Qmin] + (P[T5][Qrated] - P[T5][Qmin]) / (
            P[T5][Qmax] - P[T5][Qrated]
        ) * (
            P[Tmin][Qmax] - P[T5][Qmin]
        )  # Interpolate

        # Tmax
        P[Tmax][Qmax] = P[T47][Qmax]
        P[Tmax][Qrated] = P[T47][Qrated]
        P[Tmax][Qmin] = P[T47][Qmin]

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

def extrapolate_below(ys, y_axis, xs):
    return ys[1][y_axis] - (ys[2][y_axis] - ys[1][y_axis]) / (xs[2] - xs[1]) * (
        xs[1] - xs[0]
    )

def extrapolate_above(ys, y_axis, xs):
    i = len(xs) - 1
    i_m1 = i - 1
    i_m2 = i - 2
    return ys[i_m1][y_axis] + (ys[i_m1][y_axis] - ys[i_m2][y_axis]) / (
        xs[i_m1] - xs[i_m2]
    ) * (xs[i] - xs[i_m1])

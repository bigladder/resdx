"""
Functionality to generate EnergyPlus IDF snippets from a DXUnit object
"""

import sys
from enum import Enum

import koozie

from .conditions import CoolingConditions, HeatingConditions
from .defrost import DefrostControl, DefrostStrategy
from .dx_unit import DXUnit
from .models.nrel import NRELDXModel
from .psychrometrics import PsychState, cooling_psych_state, heating_psych_state


class IDFField:
    def __init__(self, value, name, precision=2):
        if precision is None or not isinstance(value, float):
            self.value = f"{value}"
        else:
            self.value = f"{value:.{precision}f}"
        self.name = name


def write_idf_objects(objects, output_path=None):
    if output_path is not None:
        file_handle = open(output_path, "w")
    else:
        file_handle = sys.stdout
    for obj in objects:
        print(f"{obj[0]},", file=file_handle)
        spacing = max(max([len(field.value) for field in obj[1]]) + 3, 28)
        for field in obj[1][:-1]:
            print(f"  {field.value + ',': <{spacing}}!- {field.name}", file=file_handle)
        print(
            f"  {obj[1][-1].value + ';': <{spacing}}!- {obj[1][-1].name}\n",
            file=file_handle,
        )
    if output_path is not None:
        file_handle.close()


def make_independent_variable(
    name: str, unit_type: str, rated_value: float, values: list, precision: int = 2
) -> tuple[str, list[IDFField]]:
    fields = [
        IDFField(name, "Name"),
        IDFField("Linear", "Interpolation Method"),
        IDFField("Constant", "Extrapolation Method"),
        IDFField("", "Minimum Value"),
        IDFField("", "Maximum Value"),
        IDFField(rated_value, "Normalization Reference Value", 2),
        IDFField(unit_type, "Unit Type"),
        IDFField("", "External File Name"),
        IDFField("", "External File Column Number"),
        IDFField("", "External File Starting Row Number"),
    ] + [IDFField(value, f"Value {i + 1}", precision) for i, value in enumerate(values)]

    return ("Table:IndependentVariable", fields)


def make_lookup_table(
    name: str,
    list_name: str,
    unit_type: str,
    values: list,
    precision: int = 2,
    rated_value: float | None = None,
) -> tuple[str, list[IDFField]]:
    fields = [
        IDFField(name, "Name"),
        IDFField(list_name, "Independent Variable List Name"),
        IDFField(
            "None" if rated_value is not None else "AutomaticWithDivisor",
            "Normalization Method",
        ),
        IDFField(1.0, "Normalization Divisor"),
        IDFField("", "Minimum Output"),
        IDFField("", "Maximum Output"),
        IDFField(unit_type, "Output Unit Type"),
        IDFField("", "External File Name"),
        IDFField("", "External File Column Number"),
        IDFField("", "External File Starting Row Number"),
    ] + [
        IDFField(
            value if rated_value is None else value / rated_value,
            f"Output Value {i + 1}",
            precision,
        )
        for i, value in enumerate(values)
    ]

    return ("Table:Lookup", fields)


class EnergyPlusSystemType(Enum):
    UNITARY_SYSTEM = 1
    ZONEHVAC_PTHP = 2


def write_idf(
    unit: DXUnit,
    output_path: str | None = None,
    system_name: str | None = None,
    system_type: EnergyPlusSystemType = EnergyPlusSystemType.ZONEHVAC_PTHP,
    autosize: bool = True,
    normalize: bool = True,
) -> None:
    if system_name is not None:
        system_name += " "
    else:
        system_name = ""

    objects = []

    # ------------------------------------------------------------------
    # System and Fan
    # ------------------------------------------------------------------

    if system_type == EnergyPlusSystemType.ZONEHVAC_PTHP:
        objects.append(
            (
                "ZoneHVAC:PackagedTerminalHeatPump",
                [
                    IDFField(f"{system_name}Unitary", "Name"),
                    IDFField(f"{system_name}Schedule", "Availability Schedule Name"),
                    IDFField(f"{system_name}Unitary Inlet Node", "Air Inlet Node Name"),
                    IDFField(f"{system_name}Unitary Outlet Node", "Air Outlet Node Name"),
                    IDFField("", "Outdoor Air Mixer Object Type"),
                    IDFField("", "Outdoor Air Mixer Name"),
                    IDFField("Autosize", "Cooling Supply Air Flow Rate {m3/s}"),
                    IDFField("Autosize", "Heating Supply Air Flow Rate {m3/s}"),
                    IDFField("Autosize", "No Load Supply Air Flow Rate {m3/s}"),
                    IDFField(0.0, "Cooling Outdoor Air Flow Rate {m3/s}"),
                    IDFField(0.0, "Heating Outdoor Air Flow Rate {m3/s}"),
                    IDFField(0.0, "No Load Outdoor Air Flow Rate {m3/s}"),
                    IDFField("Fan:SystemModel", "Supply Fan Object Type"),
                    IDFField(f"{system_name}Supply Fan", "Supply Fan Name"),
                    IDFField("Coil:Heating:DX:VariableSpeed", "Heating Coil Object Type"),
                    IDFField(f"{system_name}Heating Coil", "Heating Coil Name"),
                    IDFField("", "Heating Convergence Tolerance"),
                    IDFField("Coil:Cooling:DX:VariableSpeed", "Cooling Coil Object Type"),
                    IDFField(f"{system_name}Cooling Coil", "Cooling Coil Name"),
                    IDFField("", "Cooling Convergence Tolerance"),
                    IDFField("Coil:Heating:Electric", "Supplemental Heating Coil Object Type"),
                    IDFField(
                        f"{system_name}Supp Heating Coil",
                        "Supplemental Heating Coil Name",
                    ),
                    IDFField(
                        "Autosize",
                        "Maximum Supply Air Temperature from Supplemental Heater {C}",
                    ),
                    IDFField(
                        "",
                        "Maximum Outdoor Dry-Bulb Temperature for Supplemental Heater Operation {C}",
                    ),
                    IDFField("DrawThrough", "Fan Placement"),
                    IDFField(
                        f"{system_name}Fan Mode Schedule",
                        "Supply Air Fan Operating Mode Schedule Name",
                    ),
                ],
            )
        )
        objects.append(
            (
                "Coil:Heating:Electric",
                [
                    IDFField(f"{system_name}Supp Heating Coil", "Name"),
                    IDFField(f"{system_name}Schedule", "Availability Schedule Name"),
                    IDFField(1.0, "Efficiency"),
                    IDFField(0.0, "Nominal Capacity"),
                    IDFField(f"{system_name}Supply Fan Outlet Node", "Air Inlet Node Name"),
                    IDFField(f"{system_name}Unitary Outlet Node", "Air Outlet Node Name"),
                ],
            )
        )
    elif system_type == EnergyPlusSystemType.UNITARY_SYSTEM:
        objects.append(
            (
                "AirLoopHVAC:UnitarySystem",
                [
                    IDFField(f"{system_name}Unitary", "Name"),
                    IDFField("Load", "Control Type"),
                    IDFField("", "Controlling Zone or Thermostat Location"),
                    IDFField("", "Dehumidification Control Type"),
                    IDFField(f"{system_name}Schedule", "Availability Schedule Name"),
                    IDFField(f"{system_name}Unitary Inlet Node", "Air Inlet Node Name"),
                    IDFField(f"{system_name}Unitary Outlet Node", "Air Outlet Node Name"),
                    IDFField("Fan:SystemModel", "Supply Fan Object Type"),
                    IDFField(f"{system_name}Supply Fan", "Supply Fan Name"),
                    IDFField("DrawThrough", "Fan Placement"),
                    IDFField(
                        f"{system_name}Fan Mode Schedule",
                        "Supply Air Fan Operating Mode Schedule Name",
                    ),
                    IDFField(f"Coil:Heating:DX:VariableSpeed", "Heating Coil Object Type"),
                    IDFField(f"{system_name}Heating Coil", "Heating Coil Name"),
                    IDFField(
                        unit.gross_steady_state_heating_capacity() / unit.gross_total_cooling_capacity(),
                        "DX Heating Coil Sizing Ratio",
                    ),
                    IDFField("Coil:Cooling:DX:VariableSpeed", "Cooling Coil Object Type"),
                    IDFField(f"{system_name}Cooling Coil", "Cooling Coil Name"),
                    IDFField("No", "Use DOAS DX Cooling Coil"),
                    IDFField("", "Minimum Supply Air Temperature {C}"),
                    IDFField("", "Latent Load Control"),
                    IDFField("", "Supplemental Heating Coil Object Type"),
                    IDFField("", "Supplemental Heating Coil Name"),
                    IDFField("SupplyAirFlowRate", "Cooling Supply Air Flow Rate Method"),
                    IDFField("Autosize", "Cooling Supply Air Flow Rate {m3/s}"),
                    IDFField("", "Cooling Supply Air Flow Rate Per Floor Area {m3/s-m2}"),
                    IDFField(
                        "",
                        "Cooling Fraction of Autosized Cooling Supply Air Flow Rate",
                    ),
                    IDFField(
                        "",
                        "Cooling Supply Air Flow Rate Per Unit of Capacity {m3/s-W}",
                    ),
                    IDFField(f"SupplyAirFlowRate", "Heating Supply Air Flow Rate Method"),
                    IDFField("Autosize", "Heating Supply Air Flow Rate {m3/s}"),
                    IDFField("", "Heating Supply Air Flow Rate Per Floor Area {m3/s-m2}"),
                    IDFField(
                        "",
                        "Heating Fraction of Autosized Heating Supply Air Flow Rate",
                    ),
                    IDFField(
                        "",
                        "Heating Supply Air Flow Rate Per Unit of Capacity {m3/s-W}",
                    ),
                    IDFField(f"SupplyAirFlowRate", "No Load Supply Air Flow Rate Method"),
                    IDFField("Autosize", "No Load Supply Air Flow Rate {m3/s}"),
                    IDFField("", "No Load Supply Air Flow Rate Per Floor Area {m3/s-m2}"),
                    IDFField(
                        "",
                        "No Load Fraction of Autosized Cooling Supply Air Flow Rate",
                    ),
                    IDFField(
                        "",
                        "No Load Fraction of Autosized Heating Supply Air Flow Rate",
                    ),
                    IDFField(
                        "",
                        "No Load Supply Air Flow Rate Per Unit of Capacity During Cooling Operation {m3/s-W}",
                    ),
                    IDFField(
                        "",
                        "No Load Supply Air Flow Rate Per Unit of Capacity During Heating Operation {m3/s-W}",
                    ),
                    IDFField("Autosize", "Maximum Supply Air Temperature {C}"),
                    IDFField(
                        "",
                        "Maximum Outdoor Dry-Bulb Temperature for Supplemental Heater Operation {C}",
                    ),
                    IDFField("", "Outdoor Dry-Bulb Temperature Sensor Node Name"),
                    IDFField("", "Maximum Cycling Rate {cycles/hr}"),
                    IDFField("", "Heat Pump Time Constant {s}"),
                    IDFField("", "Fraction of On-Cycle Power Use"),
                    IDFField("", "Heat Pump Fan Delay Time {s}"),
                    IDFField("", "Ancilliary On-Cycle Electric Power {W}"),
                    IDFField("", "Ancilliary Off-Cycle Electric Power {W}"),
                    IDFField("", "Design Heat Recovery Water Flow Rate {m3/s}"),
                    IDFField("", "Maximum Temperature for Heat Recovery {C}"),
                    IDFField("", "Heat Recovery Water Inlet Node Name"),
                    IDFField("", "Heat Recovery Water Outlet Node Name"),
                    IDFField("", "Design Specification Multispeed Heat Pump Object Type"),
                    IDFField("", "Design Specification Multispeed Heat Pump Object Name"),
                ],
            )
        )
    else:
        raise Exception(f"Invalid EnergyPlusSystemType: {system_type}")

    objects.append(
        (
            "Schedule:Compact",
            [
                IDFField(f"{system_name}Schedule", "Name"),
                IDFField("Binary Control", "Schedule Type Limits Name"),
                IDFField("Through: 12/31", ""),
                IDFField("For: AllDays", ""),
                IDFField("Until: 24:00", ""),
                IDFField(1, "Value"),
            ],
        )
    )

    objects.append(
        (
            "Schedule:Compact",
            [
                IDFField(f"{system_name}Fan Mode Schedule", "Name"),
                IDFField("Through: 12/31", ""),
                IDFField("Fan Mode Control", "Schedule Type Limits Name"),
                IDFField("For: AllDays", ""),
                IDFField("Until: 24:00", ""),
                IDFField(0, "Value"),
            ],
        )
    )

    fan_speed_order_map = unit.fan.get_speed_order_map()
    max_fan_speed = fan_speed_order_map[-1]
    min_fan_speed = fan_speed_order_map[0]

    max_fan_speed_pressure = unit.fan.operating_pressure(max_fan_speed)

    fan_fields = [
        IDFField(f"{system_name}Supply Fan", "Name"),
        IDFField(f"{system_name}Schedule", "Availability Schedule Name"),
        IDFField(f"{system_name}Heating Coil Outlet Node", "Air Inlet Node Name"),
        IDFField(
            (
                f"{system_name}Unitary Outlet Node"
                if system_type == EnergyPlusSystemType.UNITARY_SYSTEM
                else f"{system_name}Supply Fan Outlet Node"
            ),
            "Air Outlet Node Name",
        ),
        IDFField(
            "Autosize" if autosize else unit.fan.airflow(max_fan_speed),
            "Design Maximum Air Flow Rate",
        ),
        IDFField(f"Discrete", "Speed Control Method"),
        IDFField(
            unit.fan.airflow_ratio(min_fan_speed, max_fan_speed),
            "Electric Power Minimum Flow Rate Fraction",
        ),
        IDFField(max_fan_speed_pressure, "Design Pressure Rise"),
        IDFField(1.0, "Motor Efficiency"),  # TODO: Replace if motor ever separated from fan
        IDFField(1.0, "Motor In Air Stream Fraction"),
        IDFField(
            "Autosize" if autosize else unit.fan.power(max_fan_speed),
            "Design Electric Power Consumption",
        ),
        IDFField(f"TotalEfficiencyAndPressure", "Design Power Sizing Method"),
        IDFField("", "Electric Power Per Unit Flow Rate"),
        IDFField("", "Electric Power Per Unit Flow Rate Per Unit Pressure"),
        IDFField(unit.fan.efficiency(max_fan_speed), "Fan Total Efficiency"),
        IDFField("", "Electric Power Function of Flow Fraction Curve Name"),
        IDFField(max_fan_speed_pressure, "Night Ventilation Mode Pressure Rise"),
        IDFField("", "Night Ventilation Mode Flow Fraction"),
        IDFField("", "Motor Loss Zone Name"),
        IDFField("", "Motor Loss Radiative Fraction"),
        IDFField(f"ZN-MSHP Fans", "End-Use Subcategory"),
        IDFField(unit.fan.number_of_speeds, "Number of Speeds"),
    ]
    for i, speed in enumerate(fan_speed_order_map):
        ep_speed = i + 1
        fan_speed = [
            IDFField(
                unit.fan.airflow_ratio(speed, max_fan_speed),
                f"Speed {ep_speed} Flow Fraction",
            ),
            IDFField(
                unit.fan.power_ratio(speed, max_fan_speed),
                f"Speed {ep_speed} Electric Power Fraction",
            ),
        ]

        fan_fields += fan_speed

    objects.append(("Fan:SystemModel", fan_fields))

    # ------------------------------------------------------------------
    # Independent Variable Lists
    # ------------------------------------------------------------------

    cooling_outdoor_dry_bulbs = [55.0, 82.0, 95.0, 125.0]
    cooling_indoor_wet_bulbs = [50.0, 67.0, 80.0]
    heating_outdoor_dry_bulbs = [
        koozie.to_u(unit.heating_off_temperature, "°F"),
        5.0,
        17.0,
        47.0,
        60.0,
    ]
    heating_indoor_dry_bulbs = [60.0, 70.0, 80.0]
    flow_fractions = [0.75, 1.0, 1.25]

    objects.append(
        make_independent_variable(
            f"{system_name}Cooling Outdoor Drybulb",
            "Temperature",
            koozie.convert(95.0, "°F", "°C"),
            [koozie.convert(t, "°F", "°C") for t in cooling_outdoor_dry_bulbs],
        )
    )

    objects.append(
        make_independent_variable(
            f"{system_name}Cooling Indoor Wetbulb",
            "Temperature",
            koozie.convert(67.0, "°F", "°C"),
            [koozie.convert(t, "°F", "°C") for t in cooling_indoor_wet_bulbs],
        )
    )

    objects.append(
        (
            "Table:IndependentVariableList",
            [
                IDFField(f"{system_name}Cooling fT List", "Name"),
                IDFField(
                    f"{system_name}Cooling Indoor Wetbulb",
                    "Independent Variable 1 Name",
                ),
                IDFField(
                    f"{system_name}Cooling Outdoor Drybulb",
                    "Independent Variable 2 Name",
                ),
            ],
        )
    )

    objects.append(make_independent_variable(f"{system_name}Coil Flow Fraction", "Dimensionless", 1.0, flow_fractions))

    objects.append(
        (
            "Table:IndependentVariableList",
            [
                IDFField(f"{system_name}fFF List", "Name"),
                IDFField(f"{system_name}Coil Flow Fraction", "Independent Variable 1 Name"),
            ],
        )
    )

    objects.append(
        make_independent_variable(
            f"{system_name}Heating Outdoor Drybulb",
            "Temperature",
            koozie.convert(47.0, "°F", "°C"),
            [koozie.convert(t, "°F", "°C") for t in heating_outdoor_dry_bulbs],
        )
    )

    objects.append(
        make_independent_variable(
            f"{system_name}Heating Indoor Drybulb",
            "Temperature",
            koozie.convert(70.0, "°F", "°C"),
            [koozie.convert(t, "°F", "°C") for t in heating_indoor_dry_bulbs],
        )
    )

    objects.append(
        (
            "Table:IndependentVariableList",
            [
                IDFField(f"{system_name}Heating fT List", "Name"),
                IDFField(
                    f"{system_name}Heating Indoor Drybulb",
                    "Independent Variable 1 Name",
                ),
                IDFField(
                    f"{system_name}Heating Outdoor Drybulb",
                    "Independent Variable 2 Name",
                ),
            ],
        )
    )

    # ------------------------------------------------------------------
    # Cooling
    # ------------------------------------------------------------------

    cooling_start_index = len(objects)

    cooling_coil = [
        IDFField(f"{system_name}Cooling Coil", "Name"),
        IDFField(f"{system_name}Unitary Inlet Node", "Air Inlet Node Name"),
        IDFField(f"{system_name}Cooling Coil Outlet Node", "Air Outlet Node Name"),
        IDFField(unit.number_of_cooling_speeds, "Number of Speeds"),
        IDFField(
            unit.number_of_cooling_speeds - unit.cooling_full_load_speed,
            "Nominal Speed Level",
        ),
        IDFField(
            "Autosize" if autosize else unit.gross_total_cooling_capacity(),
            "Gross Rated Total Cooling Capacity at Selected Nominal Speed Level",
            2,
        ),
        IDFField(
            "Autosize" if autosize else unit.A_full_cond.rated_volumetric_airflow,
            "Rated Air Flow Rate at Selected Nominal Speed Level",
            2,
        ),
        IDFField("", "Nominal Time for Condensate Removal to Begin"),
        IDFField(
            "",
            "Ratio of Initial Moisture Evaporation Rate and Steady State Latent Capacity",
        ),
        IDFField(f"{system_name}Cooling fPLR", "Part Load Fraction Correlation Curve Name"),
        IDFField("", "Condenser Air Inlet Node Name"),
        IDFField("", "Condenser Type"),
        IDFField("", "Evaporative Condenser Pump Rated Power Consumption"),
        IDFField(unit.crankcase_heater_capacity, "Crankcase Heater Capacity", 2),
        IDFField(
            koozie.to_u(unit.crankcase_heater_setpoint_temperature, "°C"),
            "Maximum Outdoor Dry-Bulb Temperature for Crankcase Heater Operation",
            2,
        ),
        IDFField("", "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation"),
        IDFField("", "Supply Water Storage Tank Name"),
        IDFField("", "Condensate Collection Water Storage Tank Name"),
        IDFField("", "Basin Heater Capacity"),
        IDFField("", "Basin Heater Setpoint Temperature"),
        IDFField("", "Basin Heater Operating Schedule Name"),
    ]

    objects.append(
        (
            "Curve:Linear",
            [
                IDFField(f"{system_name}Cooling fPLR", "Name"),
                IDFField(1.0 - unit.c_d_cooling, "Coefficient1 Constant"),
                IDFField(unit.c_d_cooling, "Coefficient2 x"),
                IDFField(0.0, "Minimum Value of x"),
                IDFField(1.0, "Maximum Value of x"),
            ],
        )
    )

    for speed in reversed(range(unit.number_of_cooling_speeds)):
        ep_speed = unit.number_of_cooling_speeds - speed
        condition = unit.make_condition(CoolingConditions, compressor_speed=speed)
        rated_capacity = unit.gross_total_cooling_capacity(condition)
        rated_cop = unit.gross_total_cooling_cop(condition)
        cooling_speed = [
            IDFField(
                rated_capacity,
                f"Speed {ep_speed} Reference Unit Gross Rated Total Cooling Capacity",
                1,
            ),
            IDFField(
                unit.gross_shr(condition),
                f"Speed {ep_speed} Reference Unit Gross Rated Sensible Heat Ratio",
                3,
            ),
            IDFField(rated_cop, f"Speed {ep_speed} Reference Unit Gross Rated Cooling COP", 3),
            IDFField(
                condition.rated_volumetric_airflow,
                f"Speed {ep_speed} Reference Unit Rated Air Flow Rate",
                4,
            ),
            IDFField("", f"Speed {ep_speed} Reference Unit Rated Condenser Air Flow Rate"),
            IDFField(
                "",
                f"Speed {ep_speed} Reference Unit Rated Pad Effectiveness of Evap Precooling",
            ),
            IDFField(
                f"{system_name}Cooling CapfT {ep_speed}",
                f"Speed {ep_speed} Total Cooling Capacity Function of Temperature Curve Name",
            ),
            IDFField(
                f"{system_name}Cooling CapfFF {ep_speed}",
                f"Speed {ep_speed} Total Cooling Capacity Function of Air Flow Fraction Curve Name",
            ),
            IDFField(
                f"{system_name}Cooling EIRfT {ep_speed}",
                f"Speed {ep_speed} Energy Input Ratio Function of Temperature Curve Name",
            ),
            IDFField(
                f"{system_name}Cooling EIRfFF {ep_speed}",
                f"Speed {ep_speed} Energy Input Ratio Function of Air Flow Fraction Curve Name",
            ),
        ]
        cooling_coil += cooling_speed

        capacities = []
        eirs = []
        for ff in flow_fractions:
            condition.set_mass_airflow_ratio(ff)
            capacities.append(unit.gross_total_cooling_capacity(condition))
            eirs.append(1.0 / unit.gross_total_cooling_cop(condition))

        objects.append(
            make_lookup_table(
                f"{system_name}Cooling CapfFF {ep_speed}",
                f"{system_name}fFF List",
                "Dimensionless",
                capacities,
                rated_value=rated_capacity if normalize else None,
            )
        )

        objects.append(
            make_lookup_table(
                f"{system_name}Cooling EIRfFF {ep_speed}",
                f"{system_name}fFF List",
                "Dimensionless",
                eirs,
                4,
                rated_value=1.0 / rated_cop if normalize else None,
            )
        )

        capacities = []
        eirs = []
        for t_ewb in cooling_indoor_wet_bulbs:
            for t_odb in cooling_outdoor_dry_bulbs:
                condition = unit.make_condition(
                    CoolingConditions,
                    compressor_speed=speed,
                    indoor=PsychState(
                        drybulb=koozie.fr_u(80.0, "°F"),
                        wetbulb=koozie.fr_u(t_ewb, "°F"),
                    ),
                    outdoor=cooling_psych_state(drybulb=koozie.fr_u(t_odb, "°F")),
                )
                capacities.append(unit.gross_total_cooling_capacity(condition))
                eirs.append(1.0 / unit.gross_total_cooling_cop(condition))

        objects.append(
            make_lookup_table(
                f"{system_name}Cooling CapfT {ep_speed}",
                f"{system_name}Cooling fT List",
                "Dimensionless",
                capacities,
                rated_value=rated_capacity if normalize else None,
            )
        )

        objects.append(
            make_lookup_table(
                f"{system_name}Cooling EIRfT {ep_speed}",
                f"{system_name}Cooling fT List",
                "Dimensionless",
                eirs,
                4,
                rated_value=1.0 / rated_cop if normalize else None,
            )
        )

    objects.insert(cooling_start_index, ("Coil:Cooling:DX:VariableSpeed", cooling_coil))

    # ------------------------------------------------------------------
    # Heating
    # ------------------------------------------------------------------

    heating_start_index = len(objects)

    heating_coil = [
        IDFField(f"{system_name}Heating Coil", "Name"),
        IDFField(f"{system_name}Cooling Coil Outlet Node", "Air Inlet Node Name"),
        IDFField(f"{system_name}Heating Coil Outlet Node", "Air Outlet Node Name"),
        IDFField(unit.number_of_heating_speeds, "Number of Speeds"),
        IDFField(
            unit.number_of_heating_speeds - unit.heating_full_load_speed,
            "Nominal Speed Level",
        ),
        IDFField(
            "Autosize" if autosize else unit.gross_steady_state_heating_capacity(),
            "Gross Rated Heating Capacity at Selected Nominal Speed Level",
            2,
        ),
        IDFField(
            "Autosize" if autosize else unit.H1_full_cond.rated_volumetric_airflow,
            "Rated Air Flow Rate at Selected Nominal Speed Level",
            2,
        ),
        IDFField(f"{system_name}Heating fPLR", "Part Load Fraction Correlation Curve Name"),
        IDFField(
            f"{system_name}Defrost EIR",
            "Defrost Energy Input Ratio Function of Temperature Curve Name",
        ),
        IDFField(
            koozie.to_u(unit.heating_off_temperature, "°C"),
            "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation",
            2,
        ),
        IDFField(
            koozie.to_u(unit.heating_on_temperature, "°C"),
            "Outdoor Dry-Bulb Temperature to Turn On Compressor",
            2,
        ),
        IDFField(
            (
                koozie.to_u(unit.defrost.high_temperature, "°C")
                if unit.defrost.strategy != DefrostStrategy.NONE
                else -999.0
            ),
            "Maximum Outdoor Dry-Bulb Temperature for Defrost Operation",
            2,
        ),
        IDFField(unit.crankcase_heater_capacity, "Crankcase Heater Capacity", 2),
        IDFField(
            koozie.to_u(unit.crankcase_heater_setpoint_temperature, "°C"),
            "Maximum Outdoor Dry-Bulb Temperature for Crankcase Heater Operation",
            2,
        ),
        IDFField(
            ("ReverseCycle" if unit.defrost.strategy == DefrostStrategy.REVERSE_CYCLE else "Resistive"),
            "Defrost Strategy",
        ),
        IDFField(
            "Timed" if unit.defrost.control == DefrostControl.TIMED else "OnDemand",
            "Defrost Control",
        ),
        IDFField(
            (unit.defrost.time_fraction(unit.H1_full_cond) if unit.defrost.control == DefrostControl.TIMED else ""),
            "Defrost Time Period Fraction",
            4,
        ),
        IDFField(
            (unit.defrost.resistive_power if unit.defrost.strategy == DefrostStrategy.RESISTIVE else ""),
            "Resistive Defrost Heater Capacity",
        ),
    ]

    objects.append(
        (
            "Curve:Linear",
            [
                IDFField(f"{system_name}Heating fPLR", "Name"),
                IDFField(1.0 - unit.c_d_heating, "Coefficient1 Constant"),
                IDFField(unit.c_d_heating, "Coefficient2 x"),
                IDFField(0.0, "Minimum Value of x"),
                IDFField(1.0, "Maximum Value of x"),
            ],
        )
    )

    objects.append(
        (
            "Curve:Biquadratic",
            [
                IDFField(f"{system_name}Defrost EIR", "Name"),
                IDFField(
                    1.0 / NRELDXModel.get_cooling_cop60(unit),
                    "Coefficient1 Constant",
                    4,
                ),
                IDFField(0.0, "Coefficient2 x"),
                IDFField(0.0, "Coefficient3 x**2"),
                IDFField(0.0, "Coefficient4 y"),
                IDFField(0.0, "Coefficient5 y**2"),
                IDFField(0.0, "Coefficient6 x*y"),
                IDFField(koozie.convert(50.0, "°F", "°C"), "Minimum Value of x"),
                IDFField(koozie.convert(80.0, "°F", "°C"), "Maximum Value of x"),
                IDFField(
                    koozie.to_u(unit.heating_off_temperature, "°C"),
                    "Minimum Value of y",
                ),
                IDFField(
                    koozie.to_u(unit.defrost.high_temperature, "°C"),
                    "Maximum Value of y",
                ),
            ],
        )
    )

    for speed in reversed(range(unit.number_of_heating_speeds)):
        ep_speed = unit.number_of_heating_speeds - speed
        condition = unit.make_condition(HeatingConditions, compressor_speed=speed)
        rated_capacity = unit.gross_steady_state_heating_capacity(condition)
        rated_cop = unit.gross_steady_state_heating_cop(condition)
        heating_speed = [
            IDFField(
                rated_capacity,
                f"Speed {ep_speed} Reference Unit Gross Rated Heating Capacity",
                2,
            ),
            IDFField(rated_cop, f"Speed {ep_speed} Reference Unit Gross Rated Heating COP", 3),
            IDFField(
                condition.rated_volumetric_airflow,
                f"Speed {ep_speed} Reference Unit Rated Air Flow Rate",
            ),
            IDFField(
                f"{system_name}Heating CapfT {ep_speed}",
                f"Speed {ep_speed} Heating Capacity Function of Temperature Curve Name",
            ),
            IDFField(
                f"{system_name}Heating CapfFF {ep_speed}",
                f"Speed {ep_speed} Heating Capacity Function of Air Flow Fraction Curve Name",
            ),
            IDFField(
                f"{system_name}Heating EIRfT {ep_speed}",
                f"Speed {ep_speed} Energy Input Ratio Function of Temperature Curve Name",
            ),
            IDFField(
                f"{system_name}Heating EIRfFF {ep_speed}",
                f"Speed {ep_speed} Energy Input Ratio Function of Air Flow Fraction Curve Name",
            ),
        ]
        heating_coil += heating_speed

        capacities = []
        eirs = []
        for ff in flow_fractions:
            condition.set_mass_airflow_ratio(ff)
            capacities.append(unit.gross_steady_state_heating_capacity(condition))
            eirs.append(1.0 / unit.gross_steady_state_heating_cop(condition))

        objects.append(
            make_lookup_table(
                f"{system_name}Heating CapfFF {ep_speed}",
                f"{system_name}fFF List",
                "Dimensionless",
                capacities,
                rated_value=rated_capacity if normalize else None,
            )
        )

        objects.append(
            make_lookup_table(
                f"{system_name}Heating EIRfFF {ep_speed}",
                f"{system_name}fFF List",
                "Dimensionless",
                eirs,
                4,
                rated_value=1.0 / rated_cop if normalize else None,
            )
        )

        capacities = []
        eirs = []
        heating_indoor_rh = unit.H1_full_cond.indoor.rh
        for t_edb in heating_indoor_dry_bulbs:
            for t_odb in heating_outdoor_dry_bulbs:
                condition = unit.make_condition(
                    HeatingConditions,
                    compressor_speed=speed,
                    indoor=PsychState(drybulb=koozie.fr_u(t_edb, "°F"), rel_hum=heating_indoor_rh),
                    outdoor=heating_psych_state(drybulb=koozie.fr_u(t_odb, "°F")),
                )
                capacities.append(unit.gross_steady_state_heating_capacity(condition))
                eirs.append(1.0 / unit.gross_steady_state_heating_cop(condition))

        objects.append(
            make_lookup_table(
                f"{system_name}Heating CapfT {ep_speed}",
                f"{system_name}Heating fT List",
                "Dimensionless",
                capacities,
                rated_value=rated_capacity if normalize else None,
            )
        )

        objects.append(
            make_lookup_table(
                f"{system_name}Heating EIRfT {ep_speed}",
                f"{system_name}Heating fT List",
                "Dimensionless",
                eirs,
                4,
                rated_value=1.0 / rated_cop if normalize else None,
            )
        )

    objects.insert(heating_start_index, ("Coil:Heating:DX:VariableSpeed", heating_coil))

    write_idf_objects(objects, output_path)

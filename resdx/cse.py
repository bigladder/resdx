"""
Functionality to generate CSE object snippets from a DXUnit object
"""

import sys
from enum import Enum

from koozie import fr_u, to_u

from .conditions import CoolingConditions, HeatingConditions
from .dx_unit import DXUnit, StagingType
from .psychrometrics import cooling_psych_state, heating_psych_state

INDENT = "  "


class AutoSize(Enum):
    AUTOSIZE = True


class CSEExpression:
    def __init__(self, expression: str) -> None:
        self.expression = expression

    def __str__(self):
        return self.expression


class CSEMember:
    def __init__(
        self,
        name: str,  # Name of the CSE data member
        value: str | float | int | list[float] | AutoSize | CSEExpression,  # CSE data member value
        units: str | None = None,  # Units of the CSE data member
        comment: str | None = None,  # Comment appended to the CSE data member
        precision: int = 3,  # Precision used for floating point values
        row_labels: list[str] | None = None,
        column_labels: list[str] | None = None,
        parent_object: "CSEObject | None" = None,
    ) -> None:
        self.name = name
        self.value = value
        self.precision = precision
        self.comment = comment
        self.units = units
        self.row_labels = row_labels
        self.column_labels = column_labels
        self.parent_object = parent_object

    def __str__(self):
        if self.parent_object is None:
            level = 0
        else:
            level = self.parent_object.level
        member_string = f"{INDENT * level}  {self.name} ="
        units_string = f" [{self.units}]" if self.units else ""
        comment_contents = self.comment if self.comment else ""
        comment_string = f" // {comment_contents}{units_string}" if (self.comment or self.units) else ""
        if isinstance(self.value, list):
            values = to_u(self.value, self.units) if self.units else self.value
            if self.row_labels is None and self.column_labels is None:
                value_string = ", ".join([f"{v:.{self.precision}f}" for v in values])
            else:
                assert isinstance(self.row_labels, list)
                n = len(self.value) // len(self.row_labels)  # Number of columns
                assert len(self.row_labels) * n == len(self.value)
                if self.column_labels is not None:
                    padded_labels = "  ".join([f"{label:{self.precision + 2}}" for label in self.column_labels])
                    member_string += "\n" + f"{INDENT * (level + 2)}// {padded_labels} Speed"
                value_string = ""
                for i, row_label in enumerate(self.row_labels):
                    start = i * n
                    end = n * (i + 1)
                    value_string += (
                        f"\n{INDENT * (level + 2)}"
                        + ", ".join([f"{v:.{self.precision}f}" for v in values[start:end]])
                        + f" // {row_label}"
                    )

        elif isinstance(self.value, str):
            value_string = f'"{self.value}"'
        elif isinstance(self.value, float):
            value_string = f"{(to_u(self.value, self.units) if self.units else self.value):.{self.precision}f}"
        elif isinstance(self.value, int):
            value_string = f"{self.value}"
        elif isinstance(self.value, AutoSize):
            return f"{INDENT * level}  AUTOSIZE {self.name}{comment_string}"
        elif isinstance(self.value, CSEExpression):
            value_string = str(self.value)
        else:
            raise TypeError(f"Unsupported type for CSEMember value: {type(self.value)}")
        return f"{member_string} {value_string}{comment_string}"


class CSEObject:
    def __init__(
        self, object_type: str, name: str | None, members: list[CSEMember], parent_object: "CSEObject | None" = None
    ) -> None:
        self.object_type = object_type
        self.name = name
        self.members = members
        for member in self.members:
            member.parent_object = self
        self.parent_object = parent_object
        if self.parent_object is None:
            self.level = 0
        else:
            self.level = self.parent_object.level + 1
        self.children_objects: list["CSEObject"] = []

    def add_child_object(self, child_object: "CSEObject") -> None:
        child_object.parent_object = self
        self.children_objects.append(child_object)
        child_object.level = self.level + 1

    def __str__(self):
        if len(self.members) > 0:
            member_str = "\n".join([str(member) for member in self.members])
        else:
            member_str = "\n"
        return f'{INDENT * self.level}{self.object_type} "{self.name}"\n{member_str}' + "\n\n".join(
            str(child) for child in self.children_objects
        )


class CSEPerformanceMap(CSEObject):
    def __init__(
        self,
        name: str,
        temperatures: list[float],
        reference_temperature: float,
        speeds: list[float],
        reference_speed: float,
        capacity_ratios: list[float],
        cops: list[float],
    ) -> None:
        super().__init__("PERFORMANCEMAP", name, [])
        TEMPERATURE_PRECISION = 1
        SPEED_PRECISION = 0
        self.add_child_object(
            CSEObject(
                "PMGRIDAXIS",
                f"{name} Outdoor Drybulb Temperatures",
                [
                    CSEMember("pmGXType", "Outdoor Drybulb Temperature"),
                    CSEMember("pmGXValues", temperatures, precision=TEMPERATURE_PRECISION),
                    CSEMember("pmGXRefValue", reference_temperature, precision=TEMPERATURE_PRECISION),
                ],
            )
        )
        self.add_child_object(
            CSEObject(
                "PMGRIDAXIS",
                f"{name} Speeds",
                [
                    CSEMember("pmGXType", "Speed"),
                    CSEMember("pmGXValues", speeds, precision=SPEED_PRECISION),
                    CSEMember("pmGXRefValue", reference_speed, precision=SPEED_PRECISION),
                ],
            )
        )
        temperature_labels = [f"{temperature:.0f}F" for temperature in temperatures]
        speed_labels = [f"{speed:.0f}" for speed in speeds]
        self.add_child_object(
            CSEObject(
                "PMLOOKUPDATA",
                f"{name} Capacity Ratio",
                [
                    CSEMember("pmLUType", "Capacity Ratio"),
                    CSEMember(
                        "pmLUValues",
                        capacity_ratios,
                        precision=3,
                        row_labels=temperature_labels,
                        column_labels=speed_labels,
                    ),
                ],
            )
        )
        self.add_child_object(
            CSEObject(
                "PMLOOKUPDATA",
                f"{name} COP",
                [
                    CSEMember("pmLUType", "COP"),
                    CSEMember(
                        "pmLUValues",
                        cops,
                        precision=3,
                        row_labels=temperature_labels,
                        column_labels=speed_labels,
                    ),
                ],
            )
        )


def write_cse_objects(cse_objects, output_path=None):
    if output_path is not None:
        file_handle = open(output_path, "w")
    else:
        file_handle = sys.stdout
    print("\n\n".join([str(cse_object) for cse_object in cse_objects]), file=file_handle)
    if output_path is not None:
        file_handle.close()


def write_cse(
    unit: DXUnit,
    output_path: str | None = None,
    system_name: str | None = None,
    autosize: bool = True,
) -> None:
    if system_name is None:
        system_name = "RSYS"

    objects: list[CSEObject] = []

    # Heating performance map
    heating_outdoor_dry_bulbs = [
        to_u(unit.heating_off_temperature, "°F"),
        5.0,
        17.0,
        47.0,
    ]

    heating_speeds: list[int]
    if unit.staging_type == StagingType.VARIABLE_SPEED:
        heating_speeds = [unit.heating_low_speed, unit.heating_full_load_speed, unit.heating_boost_speed]
    elif unit.staging_type == StagingType.TWO_STAGE:
        heating_speeds = [unit.heating_low_speed, unit.heating_full_load_speed]
    elif unit.staging_type == StagingType.SINGLE_STAGE:
        heating_speeds = [unit.heating_full_load_speed]
    else:
        raise ValueError(f"Unsupported staging type: {unit.staging_type}")

    capacity_ratios = []
    cops = []
    reference_capacity = unit.net_steady_state_heating_capacity()
    for t_odb in heating_outdoor_dry_bulbs:
        for speed in heating_speeds:
            condition = unit.make_condition(
                HeatingConditions,
                compressor_speed=speed,
                outdoor=heating_psych_state(drybulb=fr_u(t_odb, "°F")),
            )
            capacity_ratios.append(unit.net_steady_state_heating_capacity(condition) / reference_capacity)
            cops.append(unit.net_steady_state_heating_cop(condition))

    objects.append(
        CSEPerformanceMap(
            f"{system_name} Heating Performance Map",
            temperatures=heating_outdoor_dry_bulbs,
            reference_temperature=to_u(unit.H1_full_cond.outdoor.db, "°F"),
            speeds=[i + 1 for i in range(len(heating_speeds))],
            reference_speed=unit.H1_full_cond.compressor_speed,
            capacity_ratios=capacity_ratios,
            cops=cops,
        )
    )

    # Cooling performance map
    cooling_outdoor_dry_bulbs = [
        82.0,
        95.0,
    ]
    cooling_speeds: list[int]
    if unit.staging_type == StagingType.VARIABLE_SPEED:
        cooling_speeds = [unit.cooling_low_speed, unit.cooling_full_load_speed, unit.cooling_boost_speed]
    elif unit.staging_type == StagingType.TWO_STAGE:
        cooling_speeds = [unit.cooling_low_speed, unit.cooling_full_load_speed]
    elif unit.staging_type == StagingType.SINGLE_STAGE:
        cooling_speeds = [unit.cooling_full_load_speed]
    else:
        raise ValueError(f"Unsupported staging type: {unit.staging_type}")

    capacity_ratios = []
    cops = []
    reference_capacity = unit.net_total_cooling_capacity()
    for t_odb in cooling_outdoor_dry_bulbs:
        for speed in cooling_speeds:
            condition = unit.make_condition(
                CoolingConditions,
                compressor_speed=speed,
                outdoor=cooling_psych_state(drybulb=fr_u(t_odb, "°F")),
            )
            capacity_ratios.append(unit.net_total_cooling_capacity(condition) / reference_capacity)
            cops.append(unit.net_total_cooling_cop(condition))

    objects.append(
        CSEPerformanceMap(
            f"{system_name} Cooling Performance Map",
            temperatures=cooling_outdoor_dry_bulbs,
            reference_temperature=to_u(unit.A_full_cond.outdoor.db, "°F"),
            speeds=[i + 1 for i in range(len(cooling_speeds))],
            reference_speed=unit.A_full_cond.compressor_speed,
            capacity_ratios=capacity_ratios,
            cops=cops,
        )
    )

    objects.append(
        CSEObject(
            "RSYS",
            system_name,
            [
                CSEMember("rsType", "ASHPPM"),
                CSEMember("rsCapC", AutoSize.AUTOSIZE if autosize else unit.net_total_cooling_capacity(), "Btu/h"),
                CSEMember(
                    "rsCap47",
                    AutoSize.AUTOSIZE if autosize else unit.net_steady_state_heating_capacity(),
                    "Btu/h",
                ),
                CSEMember(
                    "rsVfPerTon",
                    unit.rated_cooling_airflow_per_rated_net_capacity[unit.cooling_full_load_speed],
                    "cfm/ton_ref",
                    precision=1,
                ),
                CSEMember("rsFanMotTy", unit.fan.fan_motor_type.name),
                CSEMember(
                    "rsFanPwrH",
                    unit.heating_fan_power() / unit.rated_heating_airflow[unit.heating_full_load_speed],
                    "W/cfm",
                    precision=3,
                ),
                CSEMember(
                    "rsFanPwrC",
                    unit.cooling_fan_power() / unit.rated_cooling_airflow[unit.cooling_full_load_speed],
                    "W/cfm",
                    precision=3,
                ),
                CSEMember("rsPerfMapHtg", f"{system_name} Heating Performance Map"),
                CSEMember("rsPerfMapClg", f"{system_name} Cooling Performance Map"),
                CSEMember(
                    "rsParElec",
                    CSEExpression(
                        f'($tdboHrAv < {to_u(unit.crankcase_heater_setpoint_temperature, "°F"):.1f}) * (1.-@RSYSRes["rsys-HVACHeatpump"].prior.H.hrsOn) * {10 * to_u(unit.net_total_cooling_capacity(), "ton_ref"):.2f}'
                    ),
                    "W",
                ),
                CSEMember("rsTypeAuxH", "RESISTANCE"),
                CSEMember("rsCtrlAuxH", "CYCLE"),
                CSEMember("rsCapAuxH", AutoSize.AUTOSIZE, "Btu/h"),
                CSEMember("rsDefrostModel", "REVCYCLEAUX"),
                CSEMember("rsASHPLockOutT", unit.heating_off_temperature, "°F", precision=1),
                CSEMember("rsCdH", unit.c_d_heating, precision=3),
                CSEMember("rsCdC", unit.c_d_cooling, precision=3),
            ],
        )
    )
    write_cse_objects(objects, output_path)

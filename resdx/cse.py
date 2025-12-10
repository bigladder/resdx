"""
Functionality to generate CSE object snippets from a DXUnit object
"""

import sys
from enum import Enum
from math import isclose

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


class CSELine:
    def __init__(self, line: str) -> None:
        self.line = line

    def __str__(self):
        return self.line


class CSEMember(CSELine):
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
                number_of_row_labels = len(self.row_labels)
                assert isinstance(self.row_labels, list)
                n = len(self.value) // number_of_row_labels  # Number of columns
                assert number_of_row_labels * n == len(self.value)
                if self.column_labels is not None:
                    padded_labels = "  ".join([f"{label:{self.precision + 2}}" for label in self.column_labels])
                    member_string += "\n" + f"{INDENT * (level + 2)}// {padded_labels} Speed"
                value_string = ""
                for i, row_label in enumerate(self.row_labels):
                    start = i * n
                    end = n * (i + 1)
                    row_label_string = f", // {row_label}"
                    if i == number_of_row_labels - 1:
                        row_label_string = f"  // {row_label}"
                    value_string += (
                        f"\n{INDENT * (level + 2)}"
                        + ", ".join([f"{v:.{self.precision}f}" for v in values[start:end]])
                        + row_label_string
                    )

        elif isinstance(self.value, str):
            if self.value[:2] == "<%" and self.value[-2:] == "%>":  # Modelkit
                value_string = self.value
            else:
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
        self, object_type: str, name: str | None, lines: list[CSELine], parent_object: "CSEObject | None" = None
    ) -> None:
        self.object_type = object_type
        self.name = name
        self.lines = lines
        for line in self.lines:
            line.parent_object = self
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
        if len(self.lines) > 0:
            line_str = "\n".join([str(line) for line in self.lines])
        else:
            line_str = "\n"
        return f'{INDENT * self.level}{self.object_type} "{self.name}"\n{line_str}' + "\n\n".join(
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


def write_cse_objects(cse_objects, output_path=None, preface: str = "") -> None:
    if output_path is not None:
        file_handle = open(output_path, "w")
    else:
        file_handle = sys.stdout
    print(preface + "\n\n".join([str(cse_object) for cse_object in cse_objects]) + "\n", file=file_handle)
    if output_path is not None:
        file_handle.close()


def write_cse(
    unit: DXUnit,
    output_path: str | None = None,
    modelkit_template: bool = False,
    system_name: str | None = None,
    autosize: bool = True,
) -> None:
    if system_name is None:
        system_name = "RSYS"

    if modelkit_template:
        system_name = "<%= system_name %>"

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
    reference_temperature = to_u(unit.H1_full_cond.outdoor.db, "°F")
    reference_speed = heating_speeds.index(unit.H1_full_cond.compressor_speed) + 1
    for t_odb in heating_outdoor_dry_bulbs:
        for speed in heating_speeds:
            condition = unit.make_condition(
                HeatingConditions,
                compressor_speed=speed,
                outdoor=heating_psych_state(drybulb=fr_u(t_odb, "°F")),
            )
            capacity = unit.net_steady_state_heating_capacity(condition)
            capacity_ratio = capacity / reference_capacity
            if isclose(t_odb, reference_temperature, abs_tol=0.1) and speed == reference_speed:
                assert capacity_ratio == 1.0, (
                    f"{t_odb:.0f}°F/{speed}: cap={capacity:.0f}, ref={reference_capacity:.0f}, full load speed= {unit.H1_full_cond.compressor_speed}, speed={heating_speeds}"
                )
            capacity_ratios.append(capacity_ratio)

            cops.append(unit.net_steady_state_heating_cop(condition))

    objects.append(
        CSEPerformanceMap(
            f"{system_name} Heating Performance Map",
            temperatures=heating_outdoor_dry_bulbs,
            reference_temperature=reference_temperature,
            speeds=[i + 1 for i in range(len(heating_speeds))],
            reference_speed=reference_speed,
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
    reference_temperature = to_u(unit.A_full_cond.outdoor.db, "°F")
    reference_speed = cooling_speeds.index(unit.A_full_cond.compressor_speed) + 1
    for t_odb in cooling_outdoor_dry_bulbs:
        for speed in cooling_speeds:
            condition = unit.make_condition(
                CoolingConditions,
                compressor_speed=speed,
                outdoor=cooling_psych_state(drybulb=fr_u(t_odb, "°F")),
            )
            capacity = unit.net_total_cooling_capacity(condition)
            capacity_ratio = capacity / reference_capacity
            if isclose(t_odb, reference_temperature, abs_tol=0.1) and speed == reference_speed:
                assert capacity_ratio == 1.0, f"{t_odb:.0f}°F/{speed}: cap={capacity:.0f}, ref={reference_capacity:.0f}"
            capacity_ratios.append(capacity_ratio)
            cops.append(unit.net_total_cooling_cop(condition))

    objects.append(
        CSEPerformanceMap(
            f"{system_name} Cooling Performance Map",
            temperatures=cooling_outdoor_dry_bulbs,
            reference_temperature=reference_temperature,
            speeds=[i + 1 for i in range(len(cooling_speeds))],
            reference_speed=reference_speed,
            capacity_ratios=capacity_ratios,
            cops=cops,
        )
    )
    cooling_heating_capacity_ratio = unit.net_total_cooling_capacity() / unit.net_steady_state_heating_capacity()

    if modelkit_template:
        cooling_capacity_lines = [
            CSELine("<% if cooling_capacity == Autosize %>"),
            CSEMember("rsCapC", AutoSize.AUTOSIZE),
            CSELine("<% else %>"),
            CSEMember("rsCapC", "<%= cooling_capacity %>", "Btu/h"),
            CSELine("<% end %>"),
        ]
        heating_capacity_lines = [
            CSELine("<% if heating_capacity == Autosize %>"),
            CSEMember("rsCap47", AutoSize.AUTOSIZE),
            CSELine("<% else %>"),
            CSEMember("rsCap47", "<%= heating_capacity %>", "Btu/h"),
            CSELine("<% end %>"),
        ]
        backup_heating_capacity_lines = [
            CSEMember("rsTypeAuxH", "<%= backup_heating_type %>"),
            CSELine('<% if backup_heating_type == "NONE" %>'),
            CSEMember("rsDefrostModel", "REVCYCLE"),
            CSELine("<% else %>"),
            CSELine("    <% if backup_heating_capacity == Autosize %>"),
            CSEMember("rsCapAuxH", AutoSize.AUTOSIZE),
            CSELine("    <% else %>"),
            CSEMember("rsCapAuxH", "<%= backup_heating_capacity %>", "Btu/h"),
            CSELine("    <% end %>"),
            CSELine('    <% if backup_heating_type == "RESISTANCE" %>'),
            CSEMember("rsCtrlAuxH", "CYCLE"),
            CSEMember("rsDefrostModel", "REVCYCLEAUX"),
            CSELine('    <% elsif backup_heating_type == "FURNACE" %>'),
            CSEMember("rsCtrlAuxH", "ALTERNATE"),
            CSEMember("rsDefrostModel", "REVCYCLEAUX"),
            CSELine("    <% end %>"),
            CSEMember("rsFxCapAuxH", 1.0, precision=1),
            CSELine("<% end %>"),
        ]
        if autosize:
            cooling_capacity_lines += [
                CSELine("<% if heating_capacity == Autosize or cooling_capacity == Autosize %>"),
                CSEMember("rsCapRat9547", cooling_heating_capacity_ratio, precision=3),
                CSELine("<% end %>"),
            ]
    else:
        cooling_capacity_lines = [
            CSEMember("rsCapC", AutoSize.AUTOSIZE if autosize else unit.net_total_cooling_capacity(), "Btu/h"),
        ]
        heating_capacity_lines = [
            CSEMember("rsCap47", AutoSize.AUTOSIZE if autosize else unit.net_steady_state_heating_capacity(), "Btu/h"),
        ]
        if unit.is_ducted:
            backup_heating_capacity_lines = [
                CSEMember("rsTypeAuxH", "RESISTANCE"),
                CSEMember("rsCapAuxH", AutoSize.AUTOSIZE),
                CSEMember("rsCtrlAuxH", "CYCLE"),
                CSEMember("rsDefrostModel", "REVCYCLEAUX"),
                CSEMember("rsFxCapAuxH", 1.0, precision=1),
            ]
        else:
            backup_heating_capacity_lines = [
                CSEMember("rsTypeAuxH", "NONE"),
                CSEMember("rsDefrostModel", "REVCYCLE"),
            ]

        if autosize:
            cooling_capacity_lines.append(CSEMember("rsCapRat9547", cooling_heating_capacity_ratio, precision=3))

    objects.append(
        CSEObject(
            "RSYS",
            system_name,
            [
                CSEMember("rsType", "ASHPPM"),
            ]
            + cooling_capacity_lines
            + [CSEMember("rsFxCapC", 1.0, precision=1)]
            + heating_capacity_lines
            + [CSEMember("rsFxCapH", 1.0, precision=1)]
            + backup_heating_capacity_lines
            + [
                CSEMember("rsHSPF", unit.hspf(), precision=1),
                CSEMember("rsSEER", unit.seer(), precision=1),
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
                        f'($tdboHrAv < {to_u(unit.crankcase_heater_setpoint_temperature, "°F"):.1f}) * (1.-@RSYSRes["{system_name}"].prior.H.hrsOn) * @RSYS["{system_name}"].capNomC * {10 / 12000.0}'
                    ),
                    "W",
                ),
                CSEMember("rsASHPLockOutT", unit.heating_off_temperature, "°F", precision=1),
                CSEMember("rsCdH", unit.c_d_heating, precision=3)
                if not modelkit_template
                else CSEMember("rsCdH", "<%= cycling_degradation_coefficient %>"),
                CSEMember("rsCdC", unit.c_d_cooling, precision=3)
                if not modelkit_template
                else CSEMember("rsCdC", "<%= cycling_degradation_coefficient %>"),
                CSEMember("rsElecMtr", "Electric Meter"),
                CSEMember("rsFuelMtr", "Gas Meter"),
            ],
        )
    )
    if modelkit_template:
        preface = (
            "<%#INITIALIZE\n"
            'parameter "system_name", :domain=>String\n'
            'parameter "cooling_capacity", :default => Autosize\n'
            'parameter "heating_capacity", :default => Autosize\n'
            f'parameter "backup_heating_type", :default => "{"RESISTANCE" if unit.is_ducted else "NONE"}", :domain => String\n'
            'parameter "backup_heating_capacity", :default => Autosize\n'
            f'parameter "cycling_degradation_coefficient", :default => {unit.c_d_heating}, :domain => Numeric\n'
            "%>\n"
            "\n"
            "<%\n"
            "if cooling_capacity == Autosize\n"
            "    if heating_capacity != Autosize\n"
            f"        cooling_capacity = heating_capacity * {cooling_heating_capacity_ratio}\n"
            "    end\n"
            "end\n"
            "\n"
            "if heating_capacity == Autosize\n"
            "    if cooling_capacity != Autosize\n"
            f"        heating_capacity = cooling_capacity / {cooling_heating_capacity_ratio}\n"
            "    end\n"
            "end\n"
            "%>\n"
            "\n"
        )
    else:
        preface = ""
    write_cse_objects(objects, output_path, preface=preface)

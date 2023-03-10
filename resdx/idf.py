import sys

import koozie

from .dx_unit import DXUnit
from .conditions import CoolingConditions, HeatingConditions
from .psychrometrics import PsychState
from .defrost import DefrostStrategy, DefrostControl
from .models.nrel import NRELDXModel

'''
Functionality to generate EnergyPlus IDF snippets from a DXUnit object
'''

class IDFField:
    def __init__(self, value, name, precision=None):
        if precision is None:
          self.value = f"{value}"
        else:
          self.value = f"{value:.{precision}f}"
        self.name = name

def write_idf_objects(objects, output_path=None):
    if output_path is not None:
        file_handle = open(output_path, 'w')
    else:
        file_handle = sys.stdout
    for obj in objects:
        print(f"{obj[0]},", file=file_handle)
        spacing = max(max([len(field.value) for field in obj[1]]) + 3, 28)
        for field in obj[1][:-1]:
            print(f"  {field.value + ',': <{spacing}}!- {field.name}", file=file_handle)
        print(f"  {obj[1][-1].value + ';': <{spacing}}!- {obj[1][-1].name}\n", file=file_handle)
    if output_path is not None:
        file_handle.close()

def make_independent_variable(name: str, unit_type: str, rated_value: float, values: list, precision: int = 2):
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
        IDFField("", "External File Starting Row Number")
    ] + [IDFField(value, f"Value {i + 1}", precision) for i, value in enumerate(values)]

    return ("Table:IndependentVariable", fields)

def make_lookup_table(name: str, list_name: str, unit_type: str, values: list, precision: int = 2):
    fields = [
        IDFField(name, "Name"),
        IDFField(list_name, "Independent Variable List Name"),
        IDFField("AutomaticWithDivisor", "Normalization Method"),
        IDFField("", "Normalization Divisor"),
        IDFField("", "Minimum Output"),
        IDFField("", "Maximum Output"),
        IDFField(unit_type, "Output Unit Type"),
        IDFField("", "External File Name"),
        IDFField("", "External File Column Number"),
        IDFField("", "External File Starting Row Number")
    ] + [IDFField(value, f"Output Value {i + 1}", precision) for i, value in enumerate(values)]

    return ("Table:Lookup", fields)

def write_idf(unit: DXUnit, output_path: str = None, system_name: str = None, autosize: bool = True):

    if system_name is not None:
        system_name += " "
    else:
        system_name = ""

    objects = []

    # ------------------------------------------------------------------
    # Independent Variable Lists
    # ------------------------------------------------------------------

    cooling_outdoor_dry_bulbs = [55., 82., 95., 125.]
    cooling_indoor_wet_bulbs = [50., 67., 80.]
    heating_outdoor_dry_bulbs = [koozie.to_u(unit.heating_off_temperature,"°F"), 5., 17., 47., 60.]
    heating_indoor_dry_bulbs = [60., 70., 80.]
    flow_fractions = [0.75,1.,1.25]

    objects.append(make_independent_variable(
        "Cooling Outdoor Drybulb",
        "Temperature",
        koozie.convert(95.,"°F","°C"),
        [koozie.convert(t,"°F","°C") for t in cooling_outdoor_dry_bulbs]
    ))

    objects.append(make_independent_variable(
        "Cooling Indoor Wetbulb",
        "Temperature",
        koozie.convert(67.,"°F","°C"),
        [koozie.convert(t,"°F","°C") for t in cooling_indoor_wet_bulbs]
    ))

    objects.append(("Table:IndependentVariableList", [
        IDFField("Cooling fT List", "Name"),
        IDFField("Cooling Indoor Wetbulb", "Independent Variable 1 Name"),
        IDFField("Cooling Outdoor Drybulb", "Independent Variable 2 Name"),
    ])),

    objects.append(make_independent_variable(
        "Coil Flow Fraction",
        "Dimensionless",
        1.,
        flow_fractions
    ))

    objects.append(("Table:IndependentVariableList", [
        IDFField("fFF List", "Name"),
        IDFField("Coil Flow Fraction", "Independent Variable 1 Name"),
    ])),

    objects.append(make_independent_variable(
        "Heating Outdoor Drybulb",
        "Temperature",
        koozie.convert(47.,"°F","°C"),
        [koozie.convert(t,"°F","°C") for t in heating_outdoor_dry_bulbs]
    ))

    objects.append(make_independent_variable(
        "Heating Indoor Drybulb",
        "Temperature",
        koozie.convert(70.,"°F","°C"),
        [koozie.convert(t,"°F","°C") for t in heating_indoor_dry_bulbs]
    ))

    objects.append(("Table:IndependentVariableList", [
        IDFField("Heating fT List", "Name"),
        IDFField("Heating Indoor Wetbulb", "Independent Variable 1 Name"),
        IDFField("Heating Outdoor Drybulb", "Independent Variable 2 Name"),
    ])),


    # ------------------------------------------------------------------
    # Cooling
    # ------------------------------------------------------------------

    cooling_coil = [
        IDFField(f"{system_name}Cooling Coil", "Name"),
        IDFField(f"{system_name}Unitary Inlet Node", "Air Inlet Node Name"),
        IDFField(f"{system_name}Cooling Coil Outlet Node", "Air Outlet Node Name"),
        IDFField(unit.number_of_cooling_speeds, "Number of Speeds"),
        IDFField(unit.number_of_cooling_speeds - unit.cooling_full_load_speed, "Nominal Speed Level"),
        IDFField("Autosize" if autosize else unit.gross_total_cooling_capacity(), "Gross Rated Total Cooling Capacity at Selected Nominal Speed Level"),
        IDFField("Autosize" if autosize else unit.A_full_cond.rated_volumetric_airflow, "Rated Air Flow Rate at Selected Nominal Speed Level"),
        IDFField("", "Nominal Time for Condensate Removal to Begin"),
        IDFField("", "Ratio of Initial Moisture Evaporation Rate and Steady State Latent Capacity"),
        IDFField(f"{system_name}Cooling fPLR", "Part Load Fraction Correlation Curve Name"),
        IDFField("", "Condenser Air Inlet Node Name"),
        IDFField("", "Condenser Type"),
        IDFField("", "Evaporative Condenser Pump Rated Power Consumption"),
        IDFField(unit.crankcase_heater_capacity, "Crankcase Heater Capacity", 2),
        IDFField(koozie.to_u(unit.crankcase_heater_setpoint_temperature,"°C"), "Maximum Outdoor Dry-Bulb Temperature for Crankcase Heater Operation", 2),
        IDFField("", "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation"),
        IDFField("", "Supply Water Storage Tank Name"),
        IDFField("", "Condensate Collection Water Storage Tank Name"),
        IDFField("", "Basin Heater Capacity"),
        IDFField("", "Basin Heater Setpoint Temperature"),
        IDFField("", "Basin Heater Operating Schedule Name")
    ]

    objects.append(("Curve:Linear", [
        IDFField(f"{system_name}Cooling fPLR", "Name"),
        IDFField(1. - unit.c_d_cooling, "Coefficient1 Constant"),
        IDFField(unit.c_d_cooling, "Coefficient2 x"),
        IDFField(0., "Minimum Value of x"),
        IDFField(1., "Maximum Value of x"),
    ])),

    for speed in reversed(range(unit.number_of_cooling_speeds)):
        ep_speed = unit.number_of_cooling_speeds - speed
        condition = unit.make_condition(CoolingConditions,compressor_speed=speed)
        cooling_speed = [
          IDFField(unit.gross_total_cooling_capacity(condition), f"Speed {ep_speed} Reference Unit Gross Rated Total Cooling Capacity", 1),
          IDFField(unit.gross_shr(condition), f"Speed {ep_speed} Reference Unit Gross Rated Sensible Heat Ratio", 3),
          IDFField(unit.gross_cooling_cop(condition), f"Speed {ep_speed} Reference Unit Gross Rated Cooling COP", 3),
          IDFField(condition.rated_volumetric_airflow, f"Speed {ep_speed} Reference Unit Rated Air Flow Rate", 4),
          IDFField("", f"Speed {ep_speed} Reference Unit Rated Condenser Air Flow Rate"),
          IDFField("", f"Speed {ep_speed} Reference Unit Rated Pad Effectiveness of Evap Precooling"),
          IDFField(f"{system_name}Cooling CapfT {ep_speed}", f"Speed {ep_speed} Total Cooling Capacity Function of Temperature Curve Name"),
          IDFField(f"{system_name}Cooling CapfFF {ep_speed}", f"Speed {ep_speed} Total Cooling Capacity Function of Air Flow Fraction Curve Name"),
          IDFField(f"{system_name}Cooling EIRfT {ep_speed}", f"Speed {ep_speed} Energy Input Ratio Function of Temperature Curve Name"),
          IDFField(f"{system_name}Cooling EIRfFF {ep_speed}", f"Speed {ep_speed} Energy Input Ratio Function of Air Flow Fraction Curve Name"),
        ]
        cooling_coil += cooling_speed

        capacities = []
        eirs = []
        for ff in flow_fractions:
            condition.set_mass_airflow_ratio(ff)
            capacities.append(unit.gross_total_cooling_capacity(condition))
            eirs.append(1./unit.gross_cooling_cop(condition))


        objects.append(make_lookup_table(
            f"{system_name}Cooling CapfFF {ep_speed}",
            "fFF List",
            "Dimensionless",
            capacities
        ))

        objects.append(make_lookup_table(
            f"{system_name}Cooling EIRfFF {ep_speed}",
            "fFF List",
            "Dimensionless",
            eirs,
            4
        ))

        capacities = []
        eirs = []
        for t_ewb in cooling_indoor_wet_bulbs:
            for t_odb in cooling_outdoor_dry_bulbs:
                condition = unit.make_condition(
                    CoolingConditions,
                    compressor_speed=speed,
                    indoor=PsychState(drybulb=koozie.fr_u(80.,"°F"),wetbulb=koozie.fr_u(t_ewb,"°F")),
                    outdoor=PsychState(drybulb=koozie.fr_u(t_odb,"°F"),rel_hum=0.4))
                capacities.append(unit.gross_total_cooling_capacity(condition))
                eirs.append(1./unit.gross_cooling_cop(condition))


        objects.append(make_lookup_table(
            f"{system_name}Cooling CapfT {ep_speed}",
            "Cooling fT List",
            "Dimensionless",
            capacities
        ))

        objects.append(make_lookup_table(
            f"{system_name}Cooling EIRfT {ep_speed}",
            "Cooling fT List",
            "Dimensionless",
            eirs,
            4
        ))


    objects.insert(0,("Coil:Cooling:DX:VariableSpeed", cooling_coil))

    # ------------------------------------------------------------------
    # Heating
    # ------------------------------------------------------------------

    heating_start_index = len(objects)

    heating_coil = [
        IDFField(f"{system_name}Heating Coil", "Name"),
        IDFField(f"{system_name}Cooling Coil Outlet Node", "Air Inlet Node Name"),
        IDFField(f"{system_name}Heating Coil Outlet Node", "Air Outlet Node Name"),
        IDFField(unit.number_of_heating_speeds, "Number of Speeds"),
        IDFField(unit.number_of_heating_speeds - unit.heating_full_load_speed, "Nominal Speed Level"),
        IDFField("Autosize" if autosize else unit.gross_steady_state_heating_capacity(), "Gross Rated Heating Capacity at Selected Nominal Speed Level"),
        IDFField("Autosize" if autosize else unit.H1_full_cond.rated_volumetric_airflow, "Rated Air Flow Rate at Selected Nominal Speed Level"),
        IDFField(f"{system_name}Heating fPLR", "Part Load Fraction Correlation Curve Name"),
        IDFField(f"{system_name}Defrost EIR", "Defrost Energy Input Ratio Function of Temperature Curve Name"),
        IDFField(koozie.to_u(unit.heating_off_temperature,"°C"), "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation", 2),
        IDFField(koozie.to_u(unit.heating_on_temperature,"°C"), "Outdoor Dry-Bulb Temperature to Turn On Compressor", 2),
        IDFField(koozie.to_u(unit.defrost.high_temperature,"°C") if unit.defrost.strategy != DefrostStrategy.NONE else -999., "Maximum Outdoor Dry-Bulb Temperature for Defrost Operation", 2),
        IDFField(unit.crankcase_heater_capacity, "Crankcase Heater Capacity", 2),
        IDFField(koozie.to_u(unit.crankcase_heater_setpoint_temperature,"°C"), "Maximum Outdoor Dry-Bulb Temperature for Crankcase Heater Operation", 2),
        IDFField("ReverseCycle" if unit.defrost.strategy == DefrostStrategy.REVERSE_CYCLE else "Resistive", "Defrost Strategy"),
        IDFField("Timed" if unit.defrost.control == DefrostControl.TIMED else "OnDemand", "Defrost Control"),
        IDFField(unit.defrost.time_fraction(unit.H1_full_cond) if unit.defrost.control == DefrostControl.TIMED else "", "Defrost Time Period Fraction", 4),
        IDFField(unit.defrost.resistive_power if unit.defrost.strategy == DefrostStrategy.RESISTIVE else "", "Resistive Defrost Heater Capacity"),
    ]

    objects.append(("Curve:Linear", [
        IDFField(f"{system_name}Heating fPLR", "Name"),
        IDFField(1. - unit.c_d_heating, "Coefficient1 Constant"),
        IDFField(unit.c_d_heating, "Coefficient2 x"),
        IDFField(0., "Minimum Value of x"),
        IDFField(1., "Maximum Value of x"),
    ]))

    objects.append(("Curve:Linear", [
        IDFField(f"{system_name}Defrost EIR", "Name"),
        IDFField(1./NRELDXModel.get_cooling_cop60(unit.model), "Coefficient1 Constant"),
        IDFField(0., "Coefficient2 x"),
    ]))

    for speed in reversed(range(unit.number_of_heating_speeds)):
        ep_speed = unit.number_of_heating_speeds - speed
        condition = unit.make_condition(HeatingConditions,compressor_speed=speed)
        heating_speed = [
          IDFField(unit.gross_steady_state_heating_capacity(condition), f"Speed {ep_speed} Reference Unit Gross Rated Heating Capacity", 2),
          IDFField(unit.gross_steady_state_heating_cop(condition), f"Speed {ep_speed} Reference Unit Gross Rated Heating COP", 3),
          IDFField(condition.rated_volumetric_airflow, f"Speed {ep_speed} Reference Unit Rated Air Flow Rate"),
          IDFField(f"{system_name}Heating CapfT {ep_speed}", f"Speed {ep_speed} Heating Capacity Function of Temperature Curve Name"),
          IDFField(f"{system_name}Heating CapfFF {ep_speed}", f"Speed {ep_speed} Heating Capacity Function of Air Flow Fraction Curve Name"),
          IDFField(f"{system_name}Heating EIRfT {ep_speed}", f"Speed {ep_speed} Energy Input Ratio Function of Temperature Curve Name"),
          IDFField(f"{system_name}Heating EIRfFF {ep_speed}", f"Speed {ep_speed} Energy Input Ratio Function of Air Flow Fraction Curve Name"),
        ]
        heating_coil += heating_speed

        capacities = []
        eirs = []
        for ff in flow_fractions:
            condition.set_mass_airflow_ratio(ff)
            capacities.append(unit.gross_steady_state_heating_capacity(condition))
            eirs.append(1./unit.gross_steady_state_heating_cop(condition))


        objects.append(make_lookup_table(
            f"{system_name}Heating CapfFF {ep_speed}",
            "fFF List",
            "Dimensionless",
            capacities
        ))

        objects.append(make_lookup_table(
            f"{system_name}Heating EIRfFF {ep_speed}",
            "fFF List",
            "Dimensionless",
            eirs,
            4
        ))

        capacities = []
        eirs = []
        for t_edb in heating_indoor_dry_bulbs:
            for t_odb in heating_outdoor_dry_bulbs:
                condition = unit.make_condition(
                    HeatingConditions,
                    compressor_speed=speed,
                    indoor=PsychState(drybulb=koozie.fr_u(t_edb,"°F"),rel_hum=0.4),
                    outdoor=PsychState(drybulb=koozie.fr_u(t_odb,"°F"),rel_hum=0.4))
                capacities.append(unit.gross_steady_state_heating_capacity(condition))
                eirs.append(1./unit.gross_steady_state_heating_cop(condition))


        objects.append(make_lookup_table(
            f"{system_name}Heating CapfT {ep_speed}",
            "Heating fT List",
            "Dimensionless",
            capacities
        ))

        objects.append(make_lookup_table(
            f"{system_name}Heating EIRfT {ep_speed}",
            "Heating fT List",
            "Dimensionless",
            eirs,
            4
        ))

    objects.insert(heating_start_index, ("Coil:Heating:DX:VariableSpeed", heating_coil))

    write_idf_objects(objects, output_path)

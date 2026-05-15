from koozie import fr_u

from resdx import (
    EnergyPlusSystemType,
    FanMotorType,
    RESNETDXModel,
    StagingType,
    get_idf_objects,
)


stage_type = "VARIABLE_SPEED"
cooling_capacity_95_W = "16426.6646052813"
heating_capacity_47_W = "16426.6646052813"
heating_capacity_17_W = "11331.763283857155"
minimum_rated_temperature_degC = "-25"
seer2 = "14.0"
eer2 = "11.870000000000001"
hspf2 = "8.3725"
motor_type = "BPM"
duct_type = "DUCTED"
heating_type = "ASHP"

# objects = get_defrost_object(system_name="HVAC")

stage_type = StagingType[stage_type]

cooling_capacity_95 = fr_u(float(cooling_capacity_95_W), "W")
heating_capacity_47 = fr_u(float(heating_capacity_47_W), "W")
heating_capacity_17 = fr_u(float(heating_capacity_17_W), "W")

minimum_rated_temperature = fr_u(float(minimum_rated_temperature_degC), "degC")

seer2: float = float(seer2)  # type: ignore
eer2: float = float(eer2)  # type: ignore
hspf2: float = float(hspf2)  # type: ignore

motor_type = FanMotorType[motor_type]

if duct_type == "DUCTED":
    is_ducted = True
else:  # duct_type == "DUCTLESS"
    is_ducted = False

if heating_type == "ASHP":
    get_heating_performance_map = True
else:  # heating_type == "GAS" or heating_type == "ELECTRIC"
    get_heating_performance_map = False

unit = RESNETDXModel(
    staging_type=stage_type,
    rated_net_total_cooling_capacity=cooling_capacity_95,
    rated_net_heating_capacity=heating_capacity_47,
    rated_net_heating_capacity_17=heating_capacity_17,
    heating_off_temperature=minimum_rated_temperature,
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
    motor_type=motor_type,
    is_ducted=is_ducted,
)

objects = get_idf_objects(
    unit=unit,
    heating_type=heating_type,
    system_name="HVAC ",
    system_type=EnergyPlusSystemType.UNITARY_SYSTEM,
    autosize=False,
    normalize=False,
    get_independent_variable_lists=True,
    get_cooling_performance_map=True,
    get_heating_performance_map=get_heating_performance_map,
)

from koozie import fr_u

from resdx import (
    EnergyPlusSystemType,
    FanMotorType,
    RESNETDXModel,
    StagingType,
    write_idf,
)


size = fr_u(38.3, "kBtu/h")
# size = fr_u(3.0, "ton_ref")
seer2 = 10.0


def get_performance_map(
    stage_type: str,
    cooling_capacity_95_btuh: str,
    heating_capacity_47_btuh: str,
    heating_capacity_17_btuh: str,
    minimum_rated_temperature_degC: str,
    seer2: str,
    eer2: str,
    hspf2: str,
    motor_type: str,
    duct_type: str,
) -> str:
    """
    Get IDF objects
    """
    stage_type = StagingType[stage_type]

    cooling_capacity_95 = fr_u(float(cooling_capacity_95_btuh), "Btu/h")
    heating_capacity_47 = fr_u(float(heating_capacity_47_btuh), "Btu/h")
    heating_capacity_17 = fr_u(float(heating_capacity_17_btuh), "Btu/h")

    minimum_rated_temperature = fr_u(float(minimum_rated_temperature_degC), "degC")

    seer2: float = float(seer2)  # type: ignore
    eer2: float = float(eer2)  # type: ignore
    hspf2: float = float(hspf2)  # type: ignore

    motor_type = FanMotorType[motor_type]

    if duct_type == "DUCTED":
        is_ducted = True
    else:  # duct_type == "DUCTLESS"
        is_ducted = False

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

    write_idf(
        unit=unit,
        output_path="HVAC1b_idf_object.idf",
        system_name=None,
        system_type=EnergyPlusSystemType.UNITARY_SYSTEM,
        autosize=False,
        normalize=False,
    )


stage_type = "SINGLE_STAGE"
cooling_capacity_95_btuh = "38300"
heating_capacity_47_btuh = "56100"
heating_capacity_17_btuh = "40000"
minimum_rated_temperature_degC = "-25"
seer2 = "12.40"
eer2 = "10.756"
hspf2 = "5.78"
motor_type = "PSC"
duct_type = "DUCTED"


get_performance_map(
    stage_type=stage_type,
    cooling_capacity_95_btuh=cooling_capacity_95_btuh,
    heating_capacity_47_btuh=heating_capacity_47_btuh,
    heating_capacity_17_btuh=heating_capacity_17_btuh,
    minimum_rated_temperature_degC=minimum_rated_temperature_degC,
    seer2=seer2,
    eer2=eer2,
    hspf2=hspf2,
    motor_type=motor_type,
    duct_type=duct_type,
)

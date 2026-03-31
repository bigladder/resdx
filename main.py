from koozie import fr_u
from resdx import RESNETDXModel, StagingType, make_neep_statistical_model_data, write_cse

size = fr_u(38.3, "kBtu/h")
# size = fr_u(3.0, "ton_ref")
seer2 = 10.0


def calculate_eer_from_seer(seer: float) -> float:
    return 10.0 + 0.84 * (seer - 11.5) if seer < 13.0 else 11.3 + 0.57 * (seer - 13.0)


def calculate_heating_capacity_17_rated(capacity_47_rated: float, stage: StagingType) -> float:
    if stage == StagingType.VARIABLE_SPEED:
        return 0.689 * capacity_47_rated
    else:  # Single or Two Speed
        return 0.626 * capacity_47_rated


tabular_data = make_neep_statistical_model_data(
    cooling_capacity_95=size,
    seer2=seer2,
    eer2=calculate_eer_from_seer(seer2),
    hspf2=10,
    heating_capacity_47=size,
    heating_capacity_17=calculate_heating_capacity_17_rated(size, stage=StagingType.VARIABLE_SPEED),
)

unit = RESNETDXModel(
    tabular_data=tabular_data,
)

cse_objects = write_cse(unit=unit, output_path="in.cse", return_text=True)

cooling_performance = "\n".join(str(cse_objects[0]))
heating_performance = "\n".join(str(cse_objects[1]))
# If no cooling or heating system, look at 301 / addendum 82 table at end of doc
print()

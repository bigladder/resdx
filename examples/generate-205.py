import resdx

# import yaml
# import cbor2
import json

import resdx.rating_solver

seer2 = 14.3
hspf2 = 7.5
size_in = resdx.fr_u(7.5, "ton_ref")


dx_unit = resdx.rating_solver.make_rating_unit(
    staging_type=resdx.StagingType.TWO_STAGE, seer=seer2, hspf=hspf2, q95=size_in
)

size = resdx.to_u(dx_unit.rated_net_total_cooling_capacity[0], "ton_ref")

dx_unit.metadata.description = (
    f"{dx_unit.number_of_cooling_speeds} speed, {size:.1f} ton, {seer2:.1f} SEER2 air conditioner"
)
dx_unit.metadata.uuid_seed = hash((dx_unit.number_of_cooling_speeds, size, seer2))
dx_unit.metadata.data_version = 2

dx_unit.fan.metadata.uuid_seed = dx_unit.metadata.uuid_seed + 1
dx_unit.fan.metadata.data_version = dx_unit.metadata.data_version


representation = dx_unit.generate_205_representation()

output_directory_path = "output"
file_name = f"residential"

# with open(f"{output_directory_path}/{file_name}.yaml", "w") as file:
#  yaml.dump(representation, file, sort_keys=False)

# with open(f"{output_directory_path}/{file_name}.cbor", "wb") as file:
#  cbor2.dump(representation, file)

with open(f"{output_directory_path}/{file_name}-unitary.RS0002.json", "w") as file:
    json.dump(representation, file, indent=4)

with open(f"{output_directory_path}/{file_name}-fan.RS0003.json", "w") as file:
    json.dump(representation["performance"]["indoor_fan_representation"], file, indent=4)

with open(f"{output_directory_path}/{file_name}-dx.RS0004.json", "w") as file:
    json.dump(representation["performance"]["dx_system_representation"], file, indent=4)

with open(f"{output_directory_path}/{file_name}-motor.RS0005.json", "w") as file:
    json.dump(
        representation["performance"]["indoor_fan_representation"]["performance"]["motor_representation"],
        file,
        indent=4,
    )

resdx.write_idf(
    dx_unit,
    f"{output_directory_path}/{file_name}-dx.RS0004.idf",
    "Heat Pump ACDXCoil 1",
    resdx.idf.EnergyPlusSystemType.UNITARY_SYSTEM,
    autosize=False,
)

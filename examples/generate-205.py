import resdx
from scipy import optimize

# import yaml
# import cbor2
import json

seer2 = 14.3
hspf2 = 7.5
cop_c, solution_c = optimize.newton(
    lambda x: resdx.DXUnit(
        number_of_cooling_speeds=2,
        rated_gross_cooling_cop=x,
        input_seer=seer2,
        rating_standard=resdx.AHRIVersion.AHRI_210_240_2023,
    ).seer()
    - seer2,
    seer2 / 3.33,
    full_output=True,
)
cop_h, solution_h = optimize.newton(
    lambda x: resdx.DXUnit(
        number_of_heating_speeds=2,
        rated_gross_heating_cop=x,
        input_hspf=hspf2,
        rating_standard=resdx.AHRIVersion.AHRI_210_240_2023,
    ).hspf()
    - hspf2,
    hspf2 / 2.0,
    full_output=True,
)
dx_unit = resdx.DXUnit(
    number_of_cooling_speeds=2,
    rated_gross_cooling_cop=cop_c,
    rated_gross_heating_cop=cop_h,
    input_seer=seer2,
    input_hspf=hspf2,
)

size = resdx.to_u(dx_unit.rated_net_total_cooling_capacity[0], "ton_ref")

dx_unit.metadata.description = f"{dx_unit.number_of_cooling_speeds} speed, {size:.1f} ton, {seer2:.1f} SEER2 air conditioner"
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
    json.dump(
        representation["performance"]["indoor_fan_representation"], file, indent=4
    )

with open(f"{output_directory_path}/{file_name}-dx.RS0004.json", "w") as file:
    json.dump(representation["performance"]["dx_system_representation"], file, indent=4)

with open(f"{output_directory_path}/{file_name}-motor.RS0005.json", "w") as file:
    json.dump(
        representation["performance"]["indoor_fan_representation"]["performance"][
            "motor_representation"
        ],
        file,
        indent=4,
    )

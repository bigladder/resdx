from typing import Dict

from koozie import fr_u

from resdx import (
    RESNETDXModel,
    make_neep_model_data,
    make_neep_statistical_model_data,
    StagingType,
    write_idf,
    EnergyPlusSystemType,
)

output_directory_path = "output"

dx_units: Dict[str, RESNETDXModel] = {}

# Mitsubishi Electric M-Series
# MUZ-GL15NAH2 AHRI Certification #: 202680596 https://ashp.neep.org/#!/product/34439/7/25000/95/7500/0///0

name = "MUZ-GL15NAH2"
Q95rated = 14000
Q47rated = 18000
seer2 = 21.0
eer2 = 13.0
hspf2 = 11.0

cooling_capacities = [
    [3428, None, 20098],  # 82
    [3100, Q95rated, 18200],  # 95
]

cooling_powers = [
    [0.19, None, 1.8],  # 82
    [0.21, 1.08, 2.0],  # 95
]

heating_capacities = [
    [2080, None, 14100],  # 5
    [2150, 12100, 16400],  # 17
    [4800, Q47rated, 20900],  # 47
]

heating_powers = [
    [0.24, None, 1.57],  # 5
    [0.2, 1.6, 2.01],  # 17
    [0.2, 1.6, 2.01],  # 47
]


dx_units[name] = RESNETDXModel(
    tabular_data=make_neep_model_data(cooling_capacities, cooling_powers, heating_capacities, heating_powers),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

dx_units[f"{name}-Statistical"] = RESNETDXModel(
    staging_type=StagingType.VARIABLE_SPEED,
    rated_net_heating_capacity=fr_u(Q47rated, "Btu/h"),
    rated_net_total_cooling_capacity=fr_u(Q95rated, "Btu/h"),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

# Mitsubishi Electric M-Series
# MUZ-FH18NAH2 AHRI Certification #: 201754303 https://ashp.neep.org/#!/product/25896/7/25000/95/7500/0///0

name = "MUZ-FH18NAH2"
Q95rated = 17200
Q47rated = 20300
seer2 = 21.0
eer2 = 12.5
hspf2 = 10.3

cooling_capacities = [
    [7126, None, 23184],  # 82
    [6450, Q95rated, 21000],  # 95
]

cooling_powers = [
    [0.37, None, 2.0],  # 82
    [0.41, 1.38, 2.22],  # 95
]

heating_capacities = [
    [3700, None, 20900],  # 5
    [4300, 12480, 24300],  # 17
    [7540, Q47rated, 30400],  # 47
]

heating_powers = [
    [0.53, None, 2.98],  # 5
    [0.6, 1.25, 3.39],  # 17
    [0.85, 1.77, 3.4],  # 47
]

dx_units[name] = RESNETDXModel(
    tabular_data=make_neep_model_data(cooling_capacities, cooling_powers, heating_capacities, heating_powers),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

dx_units[f"{name}-Statistical"] = RESNETDXModel(
    staging_type=StagingType.VARIABLE_SPEED,
    rated_net_heating_capacity=fr_u(Q47rated, "Btu/h"),
    rated_net_total_cooling_capacity=fr_u(Q95rated, "Btu/h"),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

# Mitsubishi Electric M-Series
# PUZ-HA36NHA5 AHRI Certification #: 201754321 https://ashp.neep.org/#!/product/28981/7/25000/95/7500/0///0

name = "PUZ-HA36NHA5"
Q95rated = 36000
Q47rated = 38000
seer2 = 17.0
eer2 = 12.6
hspf2 = 10.0

cooling_capacities = [
    [17000, None, 37000],  # 82
    [18000, Q95rated, 36000],  # 95
]

cooling_powers = [
    [1.0, None, 2.2],  # 82
    [1.41, 2.85, 2.85],  # 95
]

heating_capacities = [
    [8000, None, 38000],  # 5
    [13000, 28000, 38000],  # 17
    [19000, Q47rated, 40000],  # 47
]

heating_powers = [
    [0.83, None, 5.79],  # 5
    [1.14, 3.59, 5.3],  # 17
    [1.2, 3.13, 3.75],  # 47
]

dx_units[name] = RESNETDXModel(
    tabular_data=make_neep_model_data(cooling_capacities, cooling_powers, heating_capacities, heating_powers),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

dx_units[f"{name}-Statistical"] = RESNETDXModel(
    staging_type=StagingType.VARIABLE_SPEED,
    rated_net_heating_capacity=fr_u(Q47rated, "Btu/h"),
    rated_net_total_cooling_capacity=fr_u(Q95rated, "Btu/h"),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

# Mitsubishi Electric M-Series
# PUZ-A24NHA7 AHRI Certification #: 201754318 https://ashp.neep.org/#!/product/28914/7/25000/95/7500/0///0

name = "PUZ-A24NHA7"
Q95rated = 24000
Q47rated = 29000
seer2 = 24.7
eer2 = 14.3
hspf2 = 9.3

cooling_capacities = [
    [9600, None, 25900],  # 82
    [10000, Q95rated, 24000],  # 95
]

cooling_powers = [
    [0.33, None, 1.42],  # 82
    [0.43, 1.67, 1.67],  # 95
]

heating_capacities = [
    [4500, None, 13000],  # 5
    [5100, 14900, 16600],  # 17
    [9000, Q47rated, 29000],  # 47
]

heating_powers = [
    [0.4, None, 1.4],  # 5
    [0.42, 1.6, 1.9],  # 17
    [0.45, 2.07, 2.07],  # 47
]

dx_units[name] = RESNETDXModel(
    tabular_data=make_neep_model_data(cooling_capacities, cooling_powers, heating_capacities, heating_powers),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

dx_units[f"{name}-Statistical"] = RESNETDXModel(
    staging_type=StagingType.VARIABLE_SPEED,
    rated_net_heating_capacity=fr_u(Q47rated, "Btu/h"),
    rated_net_total_cooling_capacity=fr_u(Q95rated, "Btu/h"),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)


for name, unit in dx_units.items():
    print(f"Cooling info for {name}:")
    unit.print_cooling_info()
    print(f"Heating info for {name}:")
    unit.print_heating_info()
    write_idf(
        unit,
        output_path=f"{output_directory_path}/{name}.idf",
        system_name="living_unit1 ZN-MSHP",
        system_type=EnergyPlusSystemType.ZONEHVAC_PTHP,
        autosize=True,
        normalize=True,
    )
    unit.plot(f"{output_directory_path}/{name}.html")
    unit.write_validation_tables(output_directory_path, name)

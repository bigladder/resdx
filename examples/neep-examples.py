import resdx
from koozie import fr_u

output_directory_path = "output"

dx_units = {}

# Mitsubishi Electric M-Series
# MUZ-GL15NAH2 https://ashp.neep.org/#!/product/33396/7/25000///0
# SEER 21.6, EER 13, HSPF 10.8
cooling_data = resdx.VCHPDataPoints()
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(95.0,"°F"),
    capacities=[fr_u(18200,"Btu/h"),fr_u(3100,"Btu/h")],
    cops=[2.67,4.33]))
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(82.0,"°F"),
    capacities=[fr_u(20098,"Btu/h"),fr_u(3428,"Btu/h")],
    cops=[3.27,5.29]))

heating_data = resdx.VCHPDataPoints()
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(47.0,"°F"),
    capacities=[fr_u(20900,"Btu/h"),fr_u(4800,"Btu/h")],
    cops=[3.05,7.03]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(17.0,"°F"),
    capacities=[fr_u(16400,"Btu/h"),fr_u(2150,"Btu/h")],
    cops=[2.25,1.91]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(5.0,"°F"),
    capacities=[fr_u(14100,"Btu/h"),fr_u(2080,"Btu/h")],
    cops=[2.43,1.65]))

dx_units["MUZ-GL15NAH2"] = resdx.make_vchp_unit(cooling_data, heating_data, cooling_full_load_speed_ratio=14000./18200., heating_full_load_speed_ratio=18000./20900.)

# Mitsubishi Electric M-Series
# MUZ-FH18NAH2 https://ashp.neep.org/#!/product/25896/7/25000///0
# SEER 21, EER 12.5, HSPF 11
cooling_data = resdx.VCHPDataPoints()
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(95.0,"°F"),
    capacities=[fr_u(21000,"Btu/h"),fr_u(6450,"Btu/h")],
    cops=[2.77,4.61]))
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(82.0,"°F"),
    capacities=[fr_u(23184,"Btu/h"),fr_u(7126,"Btu/h")],
    cops=[3.4,5.64]))

heating_data = resdx.VCHPDataPoints()
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(47.0,"°F"),
    capacities=[fr_u(30400,"Btu/h"),fr_u(7540,"Btu/h")],
    cops=[2.62,2.6]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(17.0,"°F"),
    capacities=[fr_u(24300,"Btu/h"),fr_u(4300,"Btu/h")],
    cops=[2.1,2.1]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(5.0,"°F"),
    capacities=[fr_u(20900,"Btu/h"),fr_u(3700,"Btu/h")],
    cops=[2.06,2.05]))

dx_units["MUZ-FH18NAH2"] = resdx.make_vchp_unit(cooling_data, heating_data, cooling_full_load_speed_ratio=17200./21000., heating_full_load_speed_ratio=20300./30400.)

# Mitsubishi Electric P-Series
# PUZ-HA36NHA5 https://ashp.neep.org/#!/product/28981/7/25000///0
# SEER 17, EER 12.6, HSPF 10
cooling_data = resdx.VCHPDataPoints()
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(95.0,"°F"),
    capacities=[fr_u(36000,"Btu/h"),fr_u(18000,"Btu/h")],
    cops=[3.7,3.74]))
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(82.0,"°F"),
    capacities=[fr_u(37000,"Btu/h"),fr_u(17000,"Btu/h")],
    cops=[4.93,4.98]))

heating_data = resdx.VCHPDataPoints()
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(47.0,"°F"),
    capacities=[fr_u(40000,"Btu/h"),fr_u(19000,"Btu/h")],
    cops=[3.13,4.64]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(17.0,"°F"),
    capacities=[fr_u(38000,"Btu/h"),fr_u(13000,"Btu/h")],
    cops=[2.1,3.34]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(5.0,"°F"),
    capacities=[fr_u(38000,"Btu/h"),fr_u(8000,"Btu/h")],
    cops=[1.92,2.82]))

dx_units["PUZ-HA36NHA5"] = resdx.make_vchp_unit(cooling_data, heating_data, cooling_full_load_speed_ratio=36000./36000., heating_full_load_speed_ratio=38000./40000.)

for dx_unit in dx_units:
  print(f"Cooling info for {dx_unit}:")
  dx_units[dx_unit].print_cooling_info()
  print(f"Heating info for {dx_unit}:")
  dx_units[dx_unit].print_heating_info()
  resdx.write_idf(dx_units[dx_unit], output_path=f"{output_directory_path}/{dx_unit}.idf", system_name=dx_unit, autosize=True, )

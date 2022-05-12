import resdx

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import numpy as np

from pytest import approx


DXUnit = resdx.DXUnit
fr_u = resdx.units.fr_u
PsychState = resdx.psychrometrics.PsychState
HeatingConditions = resdx.dx_unit.HeatingConditions
CoolingConditions = resdx.dx_unit.CoolingConditions

# Single speed, SEER 13, HSPF 8
dx_unit_1_speed = DXUnit()


# Tests
def test_1_speed_regression():
  dx_unit_1_speed.print_cooling_info()
  dx_unit_1_speed.print_heating_info()
  assert dx_unit_1_speed.seer() == approx(13.0, 0.01)
  assert dx_unit_1_speed.eer() == approx(11.19, 0.01)
  assert dx_unit_1_speed.hspf() == approx(8.49, 0.01)
  assert dx_unit_1_speed.net_total_cooling_capacity() == approx(dx_unit_1_speed.net_total_cooling_capacity_rated[0],0.01)

# Two speed, SEER 17, HSPF 10
dx_unit_2_speed = DXUnit(number_of_input_stages=2)

def test_2_speed_regression():
  dx_unit_2_speed.print_cooling_info()

  dx_unit_2_speed.print_heating_info()
  dx_unit_2_speed.print_heating_info(region=2)
  assert dx_unit_2_speed.seer() == approx(14.219, 0.01)
  assert dx_unit_2_speed.eer() == approx(11.152, 0.01)
  assert dx_unit_2_speed.hspf() == approx(9.461, 0.01)
  assert dx_unit_2_speed.hspf(region=2) == approx(11.669, 0.01)

# VCHP (Fujitsu Halcyon 12) https://ashp.neep.org/#!/product/25349/7/25000///0
# SEER 21.3, EER 13.4, HSPF(4) = 11.7
cooling_data = resdx.VCHPDataPoints()
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(95.0,"°F"),
    capacities=[fr_u(13600,"Btu/hr"),fr_u(3100,"Btu/hr")],
    cops=[2.75,7.57]))
cooling_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(82.0,"°F"),
    capacities=[fr_u(15276,"Btu/hr"),fr_u(3437,"Btu/hr")],
    cops=[3.2,8.39]))

heating_data = resdx.VCHPDataPoints()
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(47.0,"°F"),
    capacities=[fr_u(19400,"Btu/hr"),fr_u(3100,"Btu/hr")],
    cops=[3.09,6.49]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(17.0,"°F"),
    capacities=[fr_u(17600,"Btu/hr"),fr_u(2824,"Btu/hr")],
    cops=[2.62,5.52]))
heating_data.append(
  resdx.VCHPDataPoint(
    drybulb=fr_u(5.0,"°F"),
    capacities=[fr_u(16710,"Btu/hr"),fr_u(2671,"Btu/hr")],
    cops=[2.37,4.89]))

vchp_unit = resdx.make_vchp_unit(cooling_data, heating_data)

def test_vchp_regression():
  assert vchp_unit.net_total_cooling_capacity() == approx(vchp_unit.net_total_cooling_capacity_rated[0],0.01)
  vchp_unit.print_cooling_info()

  vchp_unit.print_heating_info()
  vchp_unit.print_heating_info(region=2)

  assert vchp_unit.seer() == approx(20.292, 0.01)
  assert vchp_unit.eer() == approx(9.38338, 0.01)
  assert vchp_unit.hspf() == approx(20.81, 0.01)
  assert vchp_unit.hspf(region=2) == approx(22.8153, 0.01)


def test_plot():
  # Plot integrated power and capacity
  T_out = np.arange(-23,76,1)
  conditions = [dx_unit_1_speed.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(T,"°F"),rel_hum=0.4)) for T in T_out]
  Q_integrated = [dx_unit_1_speed.gross_integrated_heating_capacity(condition) for condition in conditions]
  P_integrated = [dx_unit_1_speed.gross_integrated_heating_power(condition) for condition in conditions]
  COP_integrated = [dx_unit_1_speed.gross_integrated_heating_cop(condition) for condition in conditions]

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Temp (°F)')
  ax1.set_ylabel('Capacity/Power (W)', color=color)
  ax1.plot(T_out, Q_integrated, color=color)
  ax1.plot(T_out, P_integrated, color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_ylim([0,15000])

  ax2 = ax1.twinx()

  color = 'tab:blue'
  ax2.set_ylabel('COP', color=color)
  ax2.plot(T_out, COP_integrated, color=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim([0,5.5])

  fig.tight_layout()
  plt.savefig('output/heat-pump.png')
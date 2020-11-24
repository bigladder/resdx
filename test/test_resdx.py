import resdx

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np

from pytest import approx

DXUnit = resdx.DXUnit
u = resdx.units.u
PsychState = resdx.psychrometrics.PsychState
HeatingConditions = resdx.dx_unit.HeatingConditions
CoolingConditions = resdx.dx_unit.CoolingConditions

# Single speed
dx_unit_1_speed = DXUnit()

# Two speed
dx_unit_2_speed = DXUnit(
  gross_cooling_cop_rated=[3.0,3.5],
  fan_eff_cooling_rated=[u(0.365,'W/(cu_ft/min)')]*2,
  flow_rated_per_cap_cooling_rated = [u(360.0,"(cu_ft/min)/ton_of_refrigeration"),u(300.0,"(cu_ft/min)/ton_of_refrigeration")],
  net_total_cooling_capacity_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')],
  fan_eff_heating_rated=[u(0.365,'W/(cu_ft/min)')]*2,
  gross_heating_cop_rated=[2.5, 3.0],
  flow_rated_per_cap_heating_rated = [u(360.0,"(cu_ft/min)/ton_of_refrigeration"),u(300.0,"(cu_ft/min)/ton_of_refrigeration")],
  net_heating_capacity_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')]
)

# Tests
def test_1_speed_regression():
  dx_unit_1_speed.print_cooling_info()
  dx_unit_1_speed.print_heating_info()
  assert dx_unit_1_speed.seer() == approx(9.73, 0.01)
  assert dx_unit_1_speed.eer(dx_unit_1_speed.A_full_cond) == approx(8.76, 0.01)
  assert dx_unit_1_speed.hspf() == approx(5.56, 0.01)

def test_2_speed_regression():
  dx_unit_2_speed.print_cooling_info()

  dx_unit_2_speed.print_heating_info()
  dx_unit_2_speed.print_heating_info(region=2)
  assert dx_unit_2_speed.seer() == approx(11.53, 0.01)
  assert dx_unit_2_speed.eer(dx_unit_2_speed.A_full_cond) == approx(8.76, 0.01)
  assert dx_unit_2_speed.hspf() == approx(6.18, 0.01)
  assert dx_unit_2_speed.hspf(region=2) == approx(7.78, 0.01)

def test_plot():
  # Plot integrated power and capacity
  T_out = np.arange(-23,75+1,1)
  conditions = [dx_unit_1_speed.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(T,"°F"),wetbulb=u(T-2.0,"°F"))) for T in T_out]
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

  ax2 = ax1.twinx()

  color = 'tab:blue'
  ax2.set_ylabel('COP', color=color)
  ax2.plot(T_out, COP_integrated, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.savefig('output/heat-pump.png')

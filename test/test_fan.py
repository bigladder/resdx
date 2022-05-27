import resdx
from pytest import approx

fr_u = resdx.units.fr_u
FanConditions = resdx.fan.FanConditions

def test_psc_fan():
  psc_fan = resdx.PSCFan(
    airflow_rated=[fr_u(v,'cfm') for v in [1179., 1003., 740.]],
    external_static_pressure_rated=fr_u(0., "in_H2O"),
    efficacy_rated=fr_u(0.33,'W/cfm'))

  # Open flow conditions
  assert psc_fan.airflow(FanConditions(fr_u(0., "in_H2O"),0)) == fr_u(1179.,'cfm')
  assert psc_fan.rotational_speed(FanConditions(fr_u(0., "in_H2O"),0)) == fr_u(1040.,'rpm')

  # Simple regression check
  assert psc_fan.power(FanConditions(fr_u(0.3, "in_H2O"),2)) == approx(fr_u(243.49,'W'),0.01)

  # Blocked fan
  assert psc_fan.airflow(FanConditions(fr_u(1., "in_H2O"),0)) == fr_u(0.,'cfm')
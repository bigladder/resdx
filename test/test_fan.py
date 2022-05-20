import resdx
from pytest import approx

fr_u = resdx.units.fr_u
FanConditions = resdx.fan.FanConditions

def test_psc_fan():
  psc_fan = resdx.PSCFan(
    airflow_rated=[fr_u(v,'cu_ft/min') for v in [1179., 1003., 740.]],
    external_static_pressure_rated=fr_u(0., "inch_H2O_39F"),
    efficacy_rated=fr_u(0.33,'W/(cu_ft/min)'))

  assert psc_fan.airflow(FanConditions(fr_u(0., "inch_H2O_39F"),0)) == fr_u(1179.,'cu_ft/min')
  assert psc_fan.rotational_speed(FanConditions(fr_u(0., "inch_H2O_39F"),0)) == fr_u(1040.,'rpm')
  assert psc_fan.power(FanConditions(fr_u(0.3, "inch_H2O_39F"),2)) == approx(fr_u(243.49,'W'),0.01)


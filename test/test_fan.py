import resdx
from pytest import approx

fr_u = resdx.units.fr_u

def test_psc_fan():

  design_external_static_pressure = fr_u(0.5, "in_H2O")

  psc_fan = resdx.PSCFan(
    airflow_design=[fr_u(v,'cfm') for v in [1179., 1003., 740.]],
    external_static_pressure_design=design_external_static_pressure,
    efficacy_design=fr_u(0.33,'W/cfm'))

  # Open flow conditions
  assert psc_fan.rotational_speed(0, fr_u(0., "in_H2O")) == fr_u(1040.,'rpm')

  # Design conditions
  assert psc_fan.airflow(0, design_external_static_pressure) == fr_u(1179.,'cfm')

  # Simple regression check
  assert psc_fan.power(2, fr_u(0.3, "in_H2O")) == approx(fr_u(289.08,'W'),0.01)

  # Blocked fan
  assert psc_fan.airflow(0, fr_u(1., "in_H2O")) == fr_u(0.,'cfm')

  # System curve checks
  assert psc_fan.airflow(0) == fr_u(1179.,'cfm')
  assert psc_fan.airflow(1) == approx(fr_u(1061.045,'cfm'),0.01)

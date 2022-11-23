from resdx import to_u, fr_u, VCHPDataPoint, VCHPDataPoints
from pytest import approx

'''Test file for testing unitilty functions'''

def test_units():
  assert fr_u(-40.0,"°F") == approx(fr_u(-40.0,"°C"))
  assert fr_u(32.0,"°F") == approx(fr_u(0.0,"°C"))
  assert to_u(273.15,"°C") == approx(0.0)
  assert fr_u(1.0,"in") == approx(0.0254)
  assert to_u(0.0254,"in") == approx(1.0)
  assert fr_u(3.41241633,"Btu/h") == approx(1.0,0.0001)

def test_neep_interpolation():
  cooling_data = VCHPDataPoints()
  rated_maximum_capacity = fr_u(13600,"Btu/h")
  cooling_data.append(
    VCHPDataPoint(
      drybulb=fr_u(95.0,"°F"),
      capacities=[rated_maximum_capacity,fr_u(3100,"Btu/h")],
      cops=[2.75,7.57]))
  cooling_data.append(
    VCHPDataPoint(
      drybulb=fr_u(82.0,"°F"),
      capacities=[fr_u(15276,"Btu/h"),fr_u(3437,"Btu/h")],
      cops=[3.2,8.39]))
  cooling_data.setup()
  assert cooling_data.get_capacity[0](fr_u(95.0,"°F")) == approx(rated_maximum_capacity)

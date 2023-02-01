from resdx import VCHPDataPoint, VCHPDataPoints
from pytest import approx
from koozie import fr_u

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

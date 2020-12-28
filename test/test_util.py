from resdx import to_u, fr_u
from pytest import approx

'''Test file for testing unitilty functions'''

def test_units():
  assert fr_u(-40.0,"°F") == approx(fr_u(-40.0,"°C"))
  assert fr_u(32.0,"°F") == approx(fr_u(0.0,"°C"))
  assert to_u(273.15,"°C") == approx(0.0)
  assert fr_u(1.0,"in") == approx(0.0254)
  assert to_u(0.0254,"in") == approx(1.0)
import pint # 0.15 or higher
ureg = pint.UnitRegistry()

def fr_u(value,from_units):
  '''Convert from given units into base SI units for calculation'''
  return ureg.Quantity(value, from_units).to_base_units().magnitude

#def u(value,from_units):
#  '''Convert from given units into base SI units for calculation (old function)'''
#  return fr_u(value,from_units)

def convert(value, from_units, to_units):
  '''Convert from any units to another (of the same dimension)'''
  return ureg.Quantity(value, from_units).to(to_units).magnitude

def to_u(value,to_units):
  '''Convert from base SI units to any other units'''
  base_units = ureg.Quantity(value, to_units).to_base_units().units
  return ureg.Quantity(value, base_units).to(to_units).magnitude


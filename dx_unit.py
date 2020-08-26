
#%%
import pint # 0.15 or higher
ureg = pint.UnitRegistry()

import psychrolib
psychrolib.SetUnitSystem(psychrolib.SI)

## Move to a util file?
def u(value,unit):
  return ureg.Quantity(value, unit).to_base_units().magnitude

def convert(value, from_units, to_units):
  return ureg.Quantity(value, from_units).to(to_units).magnitude

def calc_biquad(coeff, in_1, in_2):
    return coeff[0] + coeff[1] * in_1 + coeff[2] * in_1 * in_1 + coeff[3] * in_2 + coeff[4] * in_2 * in_2 + coeff[5] * in_1 * in_2

#%%
class CoolingConditions:
  def __init__(self,outdoor_drybulb=u(95.0,"°F"),
                    indoor_rh=0.4,
                    indoor_dryblub=u(80.0,"°F"),
                    press=u(1.0,"atm"),
                    fan_vol_flow_per_cap=u(350.0,"cu_ft/min/ton_of_refrigeration"),
                    compressor_speed=1.0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_dryblub
    self.press = press
    self.compressor_speed = compressor_speed

class DXUnit:

  A_cond = CoolingConditions()
  B_cond = CoolingConditions(outdoor_drybulb=82.0)

  def __init__(self,gross_total_cooling_capacity=lambda conditions : 1.0,
                    gross_sensible_cooling_capacity=lambda conditions : 1.0,
                    gross_cooling_power=lambda conditions : 1.0,
                    c_d_cooling=0.2,
                    fan_eff=u(0.365,'W/cu_ft/min'),
                    cop_rated=1.0,
                    shr_rated=0.8):
    self.gross_total_cooling_capacity = gross_total_cooling_capacity
    self.gross_sensible_cooling_capacity = gross_sensible_cooling_capacity
    self.gross_cooling_power = gross_cooling_power
    self.c_d_cooling = c_d_cooling
    self.fan_eff = fan_eff
    self.cop_rated = cop_rated
    self.shr_rated = shr_rated

  def fan_power(self):
    return

  def net_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity(conditions) #- fan_heat

  def net_cooling_power(self, conditions):
    return self.gross_cooling_power(conditions) #+ fan_power

  def eer(self, conditions):
    eer = self.cop_rated*self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)
    return eer

  def plf(self, plr):
    return 1.0 - plr*self.c_d_cooling  # eq. 11.56

  def seer(self):
    seer = self.plf(0.5) * self.eer(self.B_cond) # eq. 11.55
    return convert(seer,'','Btu/kWh')

  def writeA205(self):
    '''Write ASHRAE 205 file!!!'''
    return

def cutler_cooling_power(conditions):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  eir = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], conditions.outdoor_drybulb, T_iwb)
  cap = calc_biquad([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], conditions.outdoor_drybulb, T_iwb)
  return eir*cap

#%%
# Move this stuff to a separate

dx_unit = DXUnit(gross_cooling_power=cutler_cooling_power)

print(f"SEER: {dx_unit.seer()}")

print(f"Net power: {dx_unit.net_cooling_power(DXUnit.A_cond)}")
# %%

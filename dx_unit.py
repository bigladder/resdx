
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

def calc_quad(coeff, in_1):
    return coeff[0] + coeff[1] * in_1 + coeff[2] * in_1 * in_1 

#%%
class CoolingConditions:
  def __init__(self,outdoor_drybulb=u(95.0,"°F"),
                    indoor_rh=0.4,
                    indoor_dryblub=u(80.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=0.8, # operating flow/ rated flow
                    compressor_speed=1.0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_dryblub
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction
    self.compressor_speed = compressor_speed

class DXUnit:

  A_cond = CoolingConditions()
  B_cond = CoolingConditions(outdoor_drybulb=u(82.0,"°F"))

  def __init__(self,gross_total_cooling_capacity=lambda conditions,cap_rated : 10000,
                    gross_sensible_cooling_capacity=lambda conditions : 1.0,
                    gross_cooling_power=lambda conditions, cop_rated,cap_rated : 8000,
                    c_d_cooling=0.2,
                    fan_eff=u(0.365,'W/cu_ft/min'),
                    cop_rated=3.0,
                    flow_per_cap_rated = u(350.0,"cu_ft/min/ton_of_refrigeration"),
                    cap_rated=11000,
                    shr_rated=0.8):
    self.gross_total_cooling_capacity = gross_total_cooling_capacity # This should be in base units
    self.gross_sensible_cooling_capacity = gross_sensible_cooling_capacity
    self.gross_cooling_power = gross_cooling_power
    self.c_d_cooling = c_d_cooling
    self.fan_eff = fan_eff
    self.shr_rated = shr_rated
    self.cop_rated = cop_rated
    self.cap_rated = cap_rated
    self.flow_per_cap_rated = flow_per_cap_rated

  def fan_power(self, conditions):
    return self.fan_eff*self.standard_indoor_airflow(conditions) # eq. 11.16
  
  def fan_heat(self, conditions):
    return self.fan_power(conditions)# eq. 11.11 (in SI units)

  def standard_indoor_airflow(self,conditions):
    air_density_standard_conditions = 1.204 # in kg/m3
    return self.flow_per_cap_rated*self.cap_rated*psychrolib.GetDryAirDensity(convert(conditions.indoor_drybulb,"K","°C"), conditions.press)/air_density_standard_conditions

  def net_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity(conditions,self.cap_rated) - self.fan_heat(conditions) #- fan_heat eq. 11.3 but not considering duct losses

  def net_cooling_power(self, conditions):
    return self.gross_cooling_power(conditions,self.cop_rated,self.cap_rated) + self.fan_power(conditions) #+ fan_power eq. 11.15

  def eer(self, conditions):
    eer = self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions) # What's the output: ratio or capa/power?
    return eer

  def plf(self, plr):
    return 1.0 - plr*self.c_d_cooling  # eq. 11.56

  def seer(self):
    seer = self.plf(0.5) * self.eer(self.B_cond) # eq. 11.55
    return convert(seer,'','Btu/Wh')

  def writeA205(self):
    '''Write ASHRAE 205 file!!!'''
    return

def cutler_cooling_power(conditions,cop_rated,cap_rated):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*cap_rated*(1/cop_rated)

def cutler_cooling_capacity(conditions,cap_rated):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*cap_rated


#%%
# Move this stuff to a separate

dx_unit = DXUnit(gross_total_cooling_capacity=cutler_cooling_capacity,gross_cooling_power=cutler_cooling_power)

print(f"SEER: {dx_unit.seer()}")
print(f"Net power: {dx_unit.net_cooling_power(DXUnit.A_cond)}")
print(f"Net capacity: {dx_unit.net_total_cooling_capacity(DXUnit.A_cond)}")

# %%

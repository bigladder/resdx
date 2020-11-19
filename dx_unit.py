#%%
import sys
from enum import Enum

import numpy as np
from scipy import optimize

import math

import pint # 0.15 or higher
ureg = pint.UnitRegistry()

import psychrolib
psychrolib.SetUnitSystem(psychrolib.SI)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

## Move to a util file?
def u(value,unit):
  return ureg.Quantity(value, unit).to_base_units().magnitude

def convert(value, from_units, to_units):
  return ureg.Quantity(value, from_units).to(to_units).magnitude

def calc_biquad(coeff, in_1, in_2):
  return coeff[0] + coeff[1] * in_1 + coeff[2] * in_1 * in_1 + coeff[3] * in_2 + coeff[4] * in_2 * in_2 + coeff[5] * in_1 * in_2

def calc_quad(coeff, in_1):
  return coeff[0] + coeff[1] * in_1 + coeff[2] * in_1 * in_1

def interpolate(f, cond_1, cond_2, x):
  return f(cond_1) + (f(cond_2) - f(cond_1))/(cond_2.outdoor.db - cond_1.outdoor.db)*(x - cond_1.outdoor.db)

def find_nearest(array, value):
  closest_diff = abs(array[0] - value)
  closest_value = array[0]
  for option in array:
    diff = abs(option - value)
    if diff < closest_diff:
      closest_diff = diff
      closest_value = option
  return closest_value

#%%
class PsychState:
  def __init__(self,drybulb,pressure=u(1.0,"atm"),**kwargs):
    self.db = drybulb
    self.db_C = convert(self.db,"K","°C")
    self.p = pressure
    self.wb_set = False
    self.rh_set = False
    self.hr_set = False
    self.dp_set = False
    self.h_set = False
    self.rho_set = False
    if len(kwargs) > 1:
      sys.exit(f'only 1 can be provided')
    if "wetbulb" in kwargs:
      self.set_wb(kwargs["wetbulb"])
    elif "hum_rat" in kwargs:
      self.set_hr(kwargs["hum_rat"])
    elif "rel_hum" in kwargs:
      self.set_rh(kwargs["rel_hum"])
    elif "enthalpy" in kwargs:
      self.set_h(kwargs["enthalpy"])
    else:
      sys.exit(f'Unknonw key word argument {kwargs}.')

  def set_wb(self, wb):
    self.wb = wb
    self.wb_C = convert(self.wb,"K","°C")
    self.wb_set = True
    return self.wb

  def set_rh(self, rh):
    self.rh = rh
    if not self.wb_set:
      self.set_wb(convert(psychrolib.GetTWetBulbFromRelHum(self.db_C, self.rh, self.p),"°C","K"))
    self.rh_set = True
    return self.rh

  def set_hr(self, hr):
    self.hr = hr
    if not self.wb_set:
      self.set_wb(convert(psychrolib.GetTWetBulbFromHumRatio(self.db_C, self.hr, self.p),"°C","K"))
    self.hr_set = True
    return self.hr

  def set_h(self, h):
    self.h = h
    if not self.hr_set:
      self.set_hr(psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(self.h, self.db_C))
    self.h_set = True
    return self.h

  def get_wb(self):
    if self.wb_set:
      return self.wb
    else:
      sys.exit(f'Wetbulb not set')

  def get_wb_C(self):
    if self.wb_set:
      return self.wb_C
    else:
      sys.exit(f'Wetbulb not set')

  def set_rho(self, rho):
    self.rho = rho
    self.rho_set = True
    return self.rho

  def get_rho(self):
    if self.rho_set:
      return self.rho
    else:
      return self.set_rho(psychrolib.GetMoistAirDensity(self.db_C, self.get_hr(), self.p))

  def get_hr(self):
    if self.hr_set:
      return self.hr
    else:
      return self.set_hr(psychrolib.GetHumRatioFromTWetBulb(self.db_C, self.get_wb_C(), self.p))

  def get_h(self):
    if self.h_set:
        return self.h
    else:
      return self.set_h(psychrolib.GetMoistAirEnthalpy(self.db_C,self.get_hr()))


STANDARD_CONDITIONS = PsychState(drybulb=u(70.0,"°F"),hum_rat=0.0)

class OperatingConditions:
  def __init__(self, outdoor=STANDARD_CONDITIONS,
                     indoor=STANDARD_CONDITIONS,
                     compressor_speed=0):
    self.outdoor = outdoor
    self.indoor = indoor
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)
    self.rated_air_flow_set = False

  def set_rated_air_flow(self, air_vol_flow_per_rated_cap, capacity_rated):
    self.capacity_rated = capacity_rated
    self.std_air_vol_flow_rated = air_vol_flow_per_rated_cap*capacity_rated
    self.air_vol_flow_rated = self.std_air_vol_flow_rated*STANDARD_CONDITIONS.get_rho()/self.indoor.get_rho()
    self.air_mass_flow_rated = self.air_vol_flow_rated*self.indoor.get_rho()
    self.rated_air_flow_set = True
    self.set_air_flow(self.air_vol_flow_rated)

  def set_air_flow(self, air_vol_flow):
    self.air_vol_flow = air_vol_flow
    self.air_mass_flow = air_vol_flow*self.indoor.get_rho()
    if self.rated_air_flow_set:
      self.std_air_vol_flow = self.air_mass_flow/STANDARD_CONDITIONS.get_rho()
      self.air_mass_flow_fraction = self.air_mass_flow/self.air_mass_flow_rated
      self.std_air_vol_flow_per_capacity = self.std_air_vol_flow/self.capacity_rated

class CoolingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=u(95.0,"°F"),wetbulb=u(75.0,"°F")),
                    indoor=PsychState(drybulb=u(80.0,"°F"),wetbulb=u(67.0,"°F")),
                    compressor_speed=0):
    super().__init__(outdoor, indoor, compressor_speed)

class HeatingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=u(47.0,"°F"),wetbulb=u(43.0,"°F")),
                    indoor=PsychState(drybulb=u(70.0,"°F"),wetbulb=u(60.0,"°F")),
                    compressor_speed=0):
    super().__init__(outdoor, indoor, compressor_speed)

# AHRI 210/240 2017 distributions
class HeatingDistribution:
  outdoor_drybulbs = [u(62.0 - delta*5.0,"°F") for delta in range(18)] # 62.0 to -23 F by 5 F increments
  def __init__(self,
    outdoor_design_temperature=u(5.0,"°F"),
    fractional_hours=[0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0,0,0]
  ):
    self.outdoor_design_temperature = outdoor_design_temperature
    self.fractional_hours = fractional_hours
    self.number_of_bins = len(self.fractional_hours)
    if self.number_of_bins != 18:
      sys.exit(f'Heating distributions must be provided in 18 bins.')

class CoolingDistribution:
  outdoor_drybulbs = [u(67.0 + delta*5.0,"°F") for delta in range(8)] # 67.0 to 102 F by 5 F increments
  fractional_hours = [0.214,0.231,0.216,0.161,0.104,0.052,0.018,0.004]

  def __init__(self):
    self.number_of_bins = len(self.fractional_hours)

# Defrost characterisitcs
class DefrostControl(Enum):
  TIMED = 1,
  DEMAND = 2

class DefrostStrategy(Enum):
  REVERSE_CYCLE = 1,
  RESISTIVE = 2

class CyclingMethod(Enum):
  BETWEEN_LOW_FULL = 1
  BETWEEN_OFF_FULL = 2

class Defrost:

  def __init__(self,time_fraction = lambda conditions : u(3.5,'min')/u(60.0,'min'),
                    resistive_power = 0,
                    control = DefrostControl.TIMED,
                    strategy = DefrostStrategy.REVERSE_CYCLE,
                    high_temperature=u(41,"°F"),  # Maximum temperature for defrost operation
                    low_temperature=None,  # Minimum temperature for defrost operation
                    period=u(90,'min'),  # Time between defrost terminations (for testing)
                    max_time=u(720,"min")):  # Maximum time between defrosts allowed by controls

    # Initialize member values
    self.time_fraction = time_fraction
    self.resistive_power = resistive_power
    self.control = control
    self.strategy = strategy
    self.high_temperature = high_temperature
    self.low_temperature = low_temperature
    self.period = period
    self.max_time = max_time

    # Check inputs
    if self.strategy == DefrostStrategy.RESISTIVE and self.resistive_power <= 0:
      sys.exit(f'Defrost stratege=RESISTIVE, but resistive_power is not greater than zero.')

  def in_defrost(self, conditions):
    if self.low_temperature is not None:
      if conditions.outdoor.db > self.low_temperature and conditions.outdoor.db < self.high_temperature:
        return True
    else:
      if conditions.outdoor.db < self.high_temperature:
        return True
    return False

## Model functions

# NREL Model
'''Based on Cutler et al, but also includes internal EnergyPlus calculations'''

def cutler_cooling_power(conditions, power_scalar):
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*power_scalar

def cutler_total_cooling_capacity(conditions, capacity_scalar):
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*capacity_scalar

def cutler_steady_state_heating_power(conditions, power_scalar):
  T_idb = convert(conditions.indoor.db,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*power_scalar

def cutler_steady_state_heating_capacity(conditions, capacity_scalar):
  T_idb = convert(conditions.indoor.db,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*capacity_scalar

def epri_integrated_heating_capacity(conditions, capacity_scalar, defrost):
  # EPRI algorithm as described in EnergyPlus documentation
  if defrost.in_defrost(conditions):
    t_defrost = defrost.time_fraction(conditions)
    if defrost.control ==DefrostControl.TIMED:
        heating_capacity_multiplier = 0.909 - 107.33 * coil_diff_outdoor_air_humidity(conditions)
    else:
        heating_capacity_multiplier = 0.875 * (1-t_defrost)

    if defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
        Q_defrost_indoor_u = 0.01 * (7.222 - convert(conditions.outdoor.db,"°K","°C")) * (capacity_scalar/1.01667)
    else:
        Q_defrost_indoor_u = 0

    Q_with_frost_indoor_u = cutler_steady_state_heating_capacity(conditions,capacity_scalar) * heating_capacity_multiplier
    return Q_with_frost_indoor_u * (1-t_defrost) - Q_defrost_indoor_u * t_defrost
  else:
    return cutler_steady_state_heating_capacity(conditions,capacity_scalar)

def epri_integrated_heating_power(conditions, power_scalar, capacity_scalar, defrost):
  # EPRI algorithm as described in EnergyPlus documentation
  if defrost.in_defrost(conditions):
    t_defrost = defrost.time_fraction(conditions)
    if defrost.control == DefrostControl.TIMED:
      input_power_multiplier = 0.9 - 36.45 * coil_diff_outdoor_air_humidity(conditions)
    else:
      input_power_multiplier = 0.954 * (1-t_defrost)

    if defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
      T_iwb = convert(conditions.indoor.wb,"K","°C")
      T_odb = conditions.outdoor.db_C
      defEIRfT = calc_biquad([0.1528, 0, 0, 0, 0, 0], T_iwb, T_odb) # TODO: Check assumption from BEopt
      P_defrost = defEIRfT*(capacity_scalar/1.01667)
    else:
      P_defrost = defrost.resistive_power

    P_with_frost = cutler_steady_state_heating_power(conditions,power_scalar) * input_power_multiplier
    return P_with_frost * (1-t_defrost) + P_defrost * t_defrost
  else:
    return cutler_steady_state_heating_power(conditions,power_scalar)

def epri_defrost_time_fraction(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  return 1/(1+(0.01446/coil_diff_outdoor_air_humidity(conditions)))

def coil_diff_outdoor_air_humidity(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  T_coil_outdoor = 0.82 * convert(conditions.outdoor.db,"°K","°C") - 8.589  # In C
  saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.outdoor.p) # pressure in Pa already
  humidity_diff = conditions.outdoor.get_hr() - saturated_air_himidity_ratio
  return max(1.0e-6,humidity_diff)

def energyplus_sensible_cooling_capacity(conditions,total_capacity,bypass_factor):
  Q_t = total_capacity
  h_i = conditions.indoor.get_h()
  m_dot = conditions.air_mass_flow
  h_ADP = h_i - Q_t/(m_dot * (1-bypass_factor))
  root_fn = lambda T_ADP : psychrolib.GetSatAirEnthalpy(T_ADP, conditions.indoor.p) - h_ADP
  T_ADP = optimize.newton(root_fn, conditions.indoor.db_C)
  w_ADP = psychrolib.GetSatHumRatio(T_ADP, conditions.indoor.p)
  h_sensible = psychrolib.GetMoistAirEnthalpy(conditions.indoor.db_C,w_ADP)
  return Q_t * (h_sensible - h_ADP)/(h_i - h_ADP)

# FSEC Model

# Title 24 Model

def CA_regression(coeffs,T_ewb,T_odb,T_edb,V_std_per_rated_cap):
  return coeffs[0]*T_edb + \
    coeffs[1]*T_ewb + \
    coeffs[2]*T_odb + \
    coeffs[3]*V_std_per_rated_cap + \
    coeffs[4]*T_edb*T_odb + \
    coeffs[5]*T_edb*V_std_per_rated_cap + \
    coeffs[6]*T_ewb*T_odb + \
    coeffs[7]*T_ewb*V_std_per_rated_cap + \
    coeffs[8]*T_odb*V_std_per_rated_cap + \
    coeffs[9]*T_ewb*T_ewb + \
    coeffs[10]/V_std_per_rated_cap + \
    coeffs[11]

def title24_shr(conditions):
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Title 24 curves use °F
  T_idb = convert(conditions.indoor.db,"K","°F") # Title 24 curves use °F
  CFM_per_ton = convert(conditions.std_air_vol_flow_per_capacity,"m**3/W/s","cu_ft/min/ton_of_refrigeration")
  coeffs = [0.0242020,-0.0592153,0.0012651,0.0016375,0,0,0,-0.0000165,0,0.0002021,0,1.5085285]
  SHR = CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
  return min(1.0, SHR)

# Unified RESNET Model
resnet_cooling_power = cutler_cooling_power
resnet_total_cooling_capacity = cutler_total_cooling_capacity
resnet_sensible_cooling_capacity = energyplus_sensible_cooling_capacity
resnet_shr_rated = title24_shr
resnet_steady_state_heating_capacity = cutler_steady_state_heating_capacity
resnet_integrated_heating_capacity = epri_integrated_heating_capacity
resnet_steady_state_heating_power = cutler_steady_state_heating_power
resnet_integrated_heating_power = epri_integrated_heating_power

class DXUnit:

  regional_heating_distributions = {
    1: HeatingDistribution(u(37.0,"°F"), [0.291,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
    2: HeatingDistribution(u(27.0,"°F"), [0.215,0.189,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
    3: HeatingDistribution(u(17.0,"°F"), [0.153,0.142,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000]),
    4: HeatingDistribution(u(5.0,"°F"),  [0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0.000,0.000,0.000]),
    5: HeatingDistribution(u(-10.0,"°F"),[0.106,0.092,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001]),
    6: HeatingDistribution(u(30.0,"°F"), [0.113,0.206,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000])
    }

  cooling_distribution = CoolingDistribution()

  standard_design_heating_requirements = [(u(5000,"Btu/hr")+i*u(5000,"Btu/hr")) for i in range(0,8)] + [(u(50000,"Btu/hr")+i*u(10000,"Btu/hr")) for i in range(0,9)]

  def __init__(self,gross_total_cooling_capacity_fn=cutler_total_cooling_capacity,
                    gross_sensible_cooling_capacity_fn=resnet_sensible_cooling_capacity,
                    gross_cooling_power_fn=resnet_cooling_power,
                    shr_rated_fn=resnet_shr_rated,
                    c_d_cooling=0.2,
                    fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')],
                    gross_cooling_cop_rated=[3.0],
                    flow_rated_per_cap_cooling_rated = [u(360.0,"cu_ft/min/ton_of_refrigeration")], # TODO: Check assumption (varies by climate?)
                    net_total_cooling_capacity_rated=[u(3.0,'ton_of_refrigeration')],
                    gross_steady_state_heating_capacity_fn=resnet_steady_state_heating_capacity,
                    gross_integrated_heating_capacity_fn=resnet_integrated_heating_capacity,
                    gross_steady_state_heating_power_fn=resnet_steady_state_heating_power,
                    gross_integrated_heating_power_fn=resnet_integrated_heating_power,
                    defrost=Defrost(),
                    c_d_heating=0.2,
                    fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')],
                    gross_heating_cop_rated=[2.5],
                    flow_rated_per_cap_heating_rated = [u(360.0,"cu_ft/min/ton_of_refrigeration")], # TODO: Check assumption
                    net_heating_capacity_rated=[u(3.0,'ton_of_refrigeration')],
                    cycling_method = CyclingMethod.BETWEEN_LOW_FULL,
                    heating_off_temperature = u(10.0,"°F"), # TODO: Check value taken from Scott's script single-stage
                    heating_on_temperature = u(14.0,"°F")): # TODO: Check value taken from Scott's script single-stage

    # Initialize direct values
    self.gross_total_cooling_capacity_fn = gross_total_cooling_capacity_fn
    self.gross_sensible_cooling_capacity_fn = gross_sensible_cooling_capacity_fn
    self.gross_cooling_power_fn = gross_cooling_power_fn
    self.shr_rated_fn = shr_rated_fn
    self.c_d_cooling = c_d_cooling
    self.fan_eff_cooling_rated = fan_eff_cooling_rated
    self.gross_cooling_cop_rated = gross_cooling_cop_rated
    self.net_total_cooling_capacity_rated = net_total_cooling_capacity_rated
    self.flow_rated_per_cap_cooling_rated = flow_rated_per_cap_cooling_rated
    self.gross_steady_state_heating_capacity_fn = gross_steady_state_heating_capacity_fn
    self.gross_integrated_heating_capacity_fn = gross_integrated_heating_capacity_fn
    self.gross_steady_state_heating_power_fn = gross_steady_state_heating_power_fn
    self.gross_integrated_heating_power_fn = gross_integrated_heating_power_fn
    self.defrost = defrost
    self.c_d_heating = c_d_heating
    self.cycling_method = cycling_method
    self.fan_eff_heating_rated = fan_eff_heating_rated
    self.gross_heating_cop_rated = gross_heating_cop_rated
    self.flow_rated_per_cap_heating_rated = flow_rated_per_cap_heating_rated
    self.net_heating_capacity_rated = net_heating_capacity_rated
    self.heating_off_temperature = heating_off_temperature
    self.heating_on_temperature = heating_on_temperature

    # Initialize calculated values
    self.number_of_speeds = len(self.gross_cooling_cop_rated)
    self.gross_total_cooling_capacity_rated = [self.net_total_cooling_capacity_rated[i]*(1 + self.fan_eff_cooling_rated[i]*self.flow_rated_per_cap_cooling_rated[i]) for i in range(self.number_of_speeds)]
    self.gross_heating_capacity_rated = [self.net_heating_capacity_rated[i]*(1 + self.fan_eff_heating_rated[i]*self.flow_rated_per_cap_heating_rated[i]) for i in range(self.number_of_speeds)]
    self.bypass_factor_rated = [None]*self.number_of_speeds
    self.normalized_ntu = [None]*self.number_of_speeds

    ## Set rating conditions
    self.A_full_cond = self.make_condition(CoolingConditions)
    self.B_full_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=u(82.0,"°F"),wetbulb=u(65.0,"°F")))

    self.H1_full_cond = self.make_condition(HeatingConditions)
    self.H2_full_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(35.0,"°F"),wetbulb=u(33.0,"°F")))
    self.H3_full_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(17.0,"°F"),wetbulb=u(15.0,"°F")))

    self.shr_cooling_rated = [self.shr_rated_fn(self.A_full_cond)]
    self.calculate_bypass_factor_rated(0)

    if self.number_of_speeds > 1:
      self.A_low_cond = self.make_condition(CoolingConditions,compressor_speed=1) # Not used in AHRI ratings, only used for 'rated' SHR calculations at low speeds
      self.B_low_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=u(82.0,"°F"),wetbulb=u(65.0,"°F")),compressor_speed=1)
      self.F_low_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=u(67.0,"°F"),wetbulb=u(53.5,"°F")),compressor_speed=1)

      self.H0_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(62.0,"°F"),wetbulb=u(56.5,"°F")),compressor_speed=1)
      self.H1_low_cond = self.make_condition(HeatingConditions,compressor_speed=1)
      self.H2_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(35.0,"°F"),wetbulb=u(33.0,"°F")),compressor_speed=1)
      self.H3_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=u(17.0,"°F"),wetbulb=u(15.0,"°F")),compressor_speed=1)

      self.shr_cooling_rated += [self.shr_rated_fn(self.A_low_cond)]
      self.calculate_bypass_factor_rated(1)

    ## Check for errors

    # Check to make sure all cooling arrays are the same size if not output a warning message
    self.check_array_lengths()

    # Check to make sure arrays are in descending order
    self.check_array_order(self.net_total_cooling_capacity_rated)
    self.check_array_order(self.net_heating_capacity_rated)

  def check_array_length(self, array):
    if (len(array) != self.number_of_speeds):
      sys.exit(f'Unexpected array length ({len(array)}). Number of speeds is {self.number_of_speeds}. Array items are {array}.')

  def check_array_lengths(self):
    self.check_array_length(self.fan_eff_cooling_rated)
    self.check_array_length(self.flow_rated_per_cap_cooling_rated)
    self.check_array_length(self.net_total_cooling_capacity_rated)
    self.check_array_length(self.shr_cooling_rated)
    self.check_array_length(self.fan_eff_heating_rated)
    self.check_array_length(self.gross_heating_cop_rated)
    self.check_array_length(self.flow_rated_per_cap_heating_rated)
    self.check_array_length(self.net_heating_capacity_rated)

  def check_array_order(self, array):
    if not all(earlier >= later for earlier, later in zip(array, array[1:])):
      sys.exit(f'Arrays must be in order of decreasing capacity. Array items are {array}.')

  def make_condition(self, condition_type, compressor_speed=0, indoor=None, outdoor=None):
    if indoor is None:
      indoor = condition_type().indoor
    if outdoor is None:
      outdoor = condition_type().outdoor

    condition = condition_type(indoor=indoor, outdoor=outdoor, compressor_speed=compressor_speed)
    if condition_type == CoolingConditions:
      condition.set_rated_air_flow(self.flow_rated_per_cap_cooling_rated[compressor_speed], self.net_total_cooling_capacity_rated[compressor_speed])
    else: # if condition_type == HeatingConditions:
      condition.set_rated_air_flow(self.flow_rated_per_cap_heating_rated[compressor_speed], self.net_heating_capacity_rated[compressor_speed])
    return condition

  def fan_power(self, conditions):
    if type(conditions) == CoolingConditions:
      return self.fan_eff_cooling_rated[conditions.compressor_speed]*conditions.std_air_vol_flow # eq. 11.16
    else: # if type(conditions) == HeatingConditions:
      return self.fan_eff_heating_rated[conditions.compressor_speed]*conditions.std_air_vol_flow # eq. 11.16

  def fan_heat(self, conditions):
    return self.fan_power(conditions) # eq. 11.11 (in SI units)

  ### For cooling ###
  def gross_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity_fn(conditions,self.gross_total_cooling_capacity_rated[conditions.compressor_speed])

  def gross_sensible_cooling_capacity(self, conditions):
    return self.gross_sensible_cooling_capacity_fn(conditions,self.gross_total_cooling_capacity(conditions),self.bypass_factor(conditions))

  def gross_shr(self, conditions):
    return self.gross_sensible_cooling_capacity(conditions)/self.gross_total_cooling_capacity(conditions)

  def gross_cooling_power(self, conditions):
    return self.gross_cooling_power_fn(conditions,self.gross_total_cooling_capacity_rated[conditions.compressor_speed]/self.gross_cooling_cop_rated[conditions.compressor_speed])

  def net_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity(conditions) - self.fan_heat(conditions) # eq. 11.3 but not considering duct losses

  def net_sensible_cooling_capacity(self, conditions):
    return self.gross_sensible_cooling_capacity(conditions) - self.fan_heat(conditions)

  def net_shr(self, conditions):
    return self.net_sensible_cooling_capacity(conditions)/self.net_total_cooling_capacity(conditions)

  def net_cooling_power(self, conditions):
    return self.gross_cooling_power(conditions) + self.fan_power(conditions) # eq. 11.15

  def gross_cooling_cop(self, conditions):
    return self.gross_total_cooling_capacity(conditions)/self.gross_cooling_power(conditions)

  def net_cooling_cop(self, conditions):
    return self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)

  def get_outlet_state(self, conditions, gross_total_capacity=None, gross_sensible_capacity=None):
    if gross_total_capacity is None:
      gross_total_capacity = self.gross_total_cooling_capacity(conditions)

    if gross_sensible_capacity is None:
      gross_sensible_capacity = self.gross_sensible_cooling_capacity(conditions)

    T_idb = conditions.indoor.db
    h_i = conditions.indoor.get_h()
    m_dot_rated = conditions.air_mass_flow
    h_o = h_i - gross_total_capacity/m_dot_rated
    C_p = u(1.006,"kJ/kg/K") # Specific heat of air
    T_odb = T_idb - gross_sensible_capacity/(m_dot_rated*C_p)
    return PsychState(T_odb,pressure=conditions.indoor.p,enthalpy=h_o)

  def get_adp_state(self, inlet_state, outlet_state):
    T_idb = inlet_state.db_C
    w_i = inlet_state.get_hr()
    T_odb = outlet_state.db_C
    w_o = outlet_state.get_hr()
    root_fn = lambda T_ADP : psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, inlet_state.p) - (w_i - (w_i - w_o)/(T_idb - T_odb)*(T_idb - T_ADP))
    T_ADP = optimize.newton(root_fn, T_idb)
    w_ADP = w_i - (w_i - w_o)/(T_idb - T_odb)*(T_idb - T_ADP)
    return PsychState(convert(T_ADP,"°C","K"),pressure=inlet_state.p,hum_rat=w_ADP)

  def calculate_bypass_factor_rated(self, speed): # for rated flow rate
    if speed == 0:
      conditions = self.A_full_cond
    else:
      conditions = self.A_low_cond
    Q_s_rated = self.shr_cooling_rated[conditions.compressor_speed]*self.gross_total_cooling_capacity(conditions)
    outlet_state = self.get_outlet_state(conditions,gross_sensible_capacity=Q_s_rated)
    ADP_state = self.get_adp_state(conditions.indoor,outlet_state)
    h_i = conditions.indoor.get_h()
    h_o = outlet_state.get_h()
    h_ADP = ADP_state.get_h()
    self.bypass_factor_rated[conditions.compressor_speed] = (h_o - h_ADP)/(h_i - h_ADP)
    self.normalized_ntu[conditions.compressor_speed] = - conditions.air_mass_flow * math.log(self.bypass_factor_rated[conditions.compressor_speed]) # A0 = - m_dot * ln(BF)

  def bypass_factor(self,conditions):
    return math.exp(-self.normalized_ntu[conditions.compressor_speed]/conditions.air_mass_flow)

  def get_adp_state_from_conditions(self, conditions):
    outlet_state = self.get_outlet_state(conditions)
    return self.get_adp_state(conditions.indoor,outlet_state)

  def eer(self, conditions):
    return convert(self.net_cooling_cop(conditions),'','Btu/Wh')

  def seer(self):
    if self.number_of_speeds == 1:
      plf = 1.0 - 0.5*self.c_d_cooling # eq. 11.56
      seer = plf*self.net_cooling_cop(self.B_full_cond) # eq. 11.55 (using COP to keep things in SI units for now)
    else:  #elif self.number_of_speeds == 2:
      sizing_factor = 1.1 # eq. 11.61
      q_sum = 0.0
      e_sum = 0.0
      for i in range(self.cooling_distribution.number_of_bins):
        t = self.cooling_distribution.outdoor_drybulbs[i]
        n = self.cooling_distribution.fractional_hours[i]
        bl = (t - u(65.0,"°F"))/(u(95,"°F") - u(65.0,"°F"))*self.net_total_cooling_capacity(self.A_full_cond)/sizing_factor # eq. 11.60
        q_low = interpolate(self.net_total_cooling_capacity, self.F_low_cond, self.B_low_cond, t) # eq. 11.62
        p_low = interpolate(self.net_cooling_power, self.F_low_cond, self.B_low_cond, t) # eq. 11.63
        q_full = interpolate(self.net_total_cooling_capacity, self.B_full_cond, self.A_full_cond, t) # eq. 11.64
        p_full = interpolate(self.net_cooling_power, self.B_full_cond, self.A_full_cond, t) # eq. 11.65
        if bl <= q_low:
          clf_low = bl/q_low # eq. 11.68
          plf_low = 1.0 - self.c_d_cooling*(1.0 - clf_low) # eq. 11.69
          q = clf_low*q_low*n # eq. 11.66
          e = clf_low*p_low*n/plf_low # eq. 11.67
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL:
          clf_low = (q_full - bl)/(q_full - q_low) # eq. 11.74
          clf_full = 1.0 - clf_low # eq. 11.75
          q = (clf_low*q_low + clf_full*q_full)*n # eq. 11.72
          e = (clf_low*p_low + clf_full*p_full)*n # eq. 11.73
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL:
          clf_full = bl/q_full # eq. 11.78
          plf_full = 1.0 - self.c_d_cooling*(1.0 - clf_full) # eq. 11.79
          q = clf_full*q_full*n # eq. 11.76
          e = clf_full*p_full*n/plf_full # eq. 11.77
        else: # elif bl >= q_full
          q = q_full*n
          e = p_full*n
        q_sum += q
        e_sum += e

      seer = q_sum/e_sum # e.q. 11.59
    return convert(seer,'','Btu/Wh')

  ### For heating ###
  def gross_steady_state_heating_capacity(self, conditions):
    return self.gross_steady_state_heating_capacity_fn(conditions, self.gross_heating_capacity_rated[conditions.compressor_speed])

  def gross_steady_state_heating_power(self, conditions):
    return self.gross_steady_state_heating_power_fn(conditions, self.gross_heating_capacity_rated[conditions.compressor_speed]/self.gross_heating_cop_rated[conditions.compressor_speed])

  def gross_integrated_heating_capacity(self, conditions):
    return self.gross_integrated_heating_capacity_fn(conditions, self.gross_heating_capacity_rated[conditions.compressor_speed], self.defrost)

  def gross_integrated_heating_power(self, conditions):
    return self.gross_integrated_heating_power_fn(conditions, self.gross_heating_capacity_rated[conditions.compressor_speed]/self.gross_heating_cop_rated[conditions.compressor_speed], self.gross_heating_capacity_rated[conditions.compressor_speed], self.defrost)

  def net_steady_state_heating_capacity(self, conditions):
    return self.gross_steady_state_heating_capacity(conditions) + self.fan_heat(conditions) # eq. 11.31

  def net_steady_state_heating_power(self, conditions):
    return self.gross_steady_state_heating_power(conditions) + self.fan_power(conditions) # eq. 11.41

  def net_integrated_heating_capacity(self, conditions):
    return self.gross_integrated_heating_capacity(conditions) + self.fan_heat(conditions) # eq. 11.31

  def net_integrated_heating_power(self, conditions):
    return self.gross_integrated_heating_power(conditions) + self.fan_power(conditions) # eq. 11.41

  def gross_steady_state_heating_cop(self, conditions):
    return self.gross_steady_state_heating_capacity(conditions)/self.gross_steady_state_heating_power(conditions)

  def gross_integrated_heating_cop(self, conditions):
    return self.gross_integrated_heating_capacity(conditions)/self.gross_integrated_heating_power(conditions)

  def net_steady_state_heating_cop(self, conditions):
    return self.net_steady_state_heating_capacity(conditions)/self.net_steady_state_heating_power(conditions)

  def net_integrated_heating_cop(self, conditions):
    return self.net_integrated_heating_capacity(conditions)/self.net_integrated_heating_power(conditions)

  def hspf(self, climate_region=4):
    q_sum = 0.0
    e_sum = 0.0
    rh_sum = 0.0

    c = 0.77 # eq. 11.110 (agreement factor)
    t_od = self.regional_heating_distributions[climate_region].outdoor_design_temperature

    dhr_min = self.net_integrated_heating_capacity(self.H1_full_cond)*(u(65,"°F")-t_od)/(u(60,"°R")) # eq. 11.111
    dhr_min = find_nearest(self.standard_design_heating_requirements, dhr_min)

    for i in range(self.regional_heating_distributions[climate_region].number_of_bins):
      t = self.regional_heating_distributions[climate_region].outdoor_drybulbs[i]
      n = self.regional_heating_distributions[climate_region].fractional_hours[i]
      bl = (u(65,"°F")-t)/(u(65,"°F")-t_od)*c*dhr_min # eq. 11.109

      t_ob = u(45,"°F") # eq. 11.119
      if t >= t_ob or t <= u(17,"°F"):
        q_full = interpolate(self.net_integrated_heating_capacity, self.H3_full_cond, self.H1_full_cond, t) # eq. 11.117
        p_full = interpolate(self.net_integrated_heating_power, self.H3_full_cond, self.H1_full_cond, t) # eq. 11.117
      else: # elif t > u(17,"°F") and t < t_ob
        q_full = interpolate(self.net_integrated_heating_capacity, self.H3_full_cond, self.H2_full_cond, t) # eq. 11.118
        p_full = interpolate(self.net_integrated_heating_power, self.H3_full_cond, self.H2_full_cond, t) # eq. 11.117
      cop_full = q_full/p_full

      if t <= self.heating_off_temperature or cop_full < 1.0:
        delta_full = 0.0 # eq. 11.120 & 11.159
      elif t > self.heating_on_temperature and cop_full >= 1.0:
        delta_full = 1.0 # eq. 11.122 & 11.160
      else:
        delta_full = 0.5 # eq. 11.121 & 11.161

      if q_full > bl:
        hlf_full = bl/q_full # eq. 11.115 & 11.154
      else:
        hlf_full = 1.0 # eq. 11.116

      if self.number_of_speeds == 1:
        plf_full = 1.0 - self.c_d_heating*(1.0 - hlf_full) # eq. 11.125
        e = p_full*hlf_full*delta_full*n/plf_full # eq. 11.156 (not shown for single stage)
        rh = (bl - q_full*hlf_full*delta_full)*n # eq. 11.126
      else: # elif self.number_of_speeds == 2:
        t_ob = u(40,"°F") # eq. 11.134
        if t >= t_ob:
          q_low = interpolate(self.net_integrated_heating_capacity, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.135
          p_low = interpolate(self.net_integrated_heating_power, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.138
        elif t <= u(17.0,"°F"):
          q_low = interpolate(self.net_integrated_heating_capacity, self.H1_low_cond, self.H3_low_cond, t) # eq. 11.137
          p_low = interpolate(self.net_integrated_heating_power, self.H1_low_cond, self.H3_low_cond, t) # eq. 11.140
        else:
          q_low = interpolate(self.net_integrated_heating_capacity, self.H2_low_cond, self.H3_low_cond, t) # eq. 11.136
          p_low = interpolate(self.net_integrated_heating_power, self.H2_low_cond, self.H3_low_cond, t) # eq. 11.139

        cop_low = q_low/p_low
        if t <= self.heating_off_temperature or cop_low < 1.0:
          delta_low = 0.0 # eq. 11.147
        elif t > self.heating_on_temperature and cop_low >= 1.0:
          delta_low = 1.0 # eq. 11.149
        else:
          delta_low = 0.5 # eq. 11.148

        if bl <= q_low:
          hlf_low = bl/q_low # eq. 11.143
          plf_low = 1.0 - self.c_d_heating*(1.0 - hlf_low) # eq. 11.144
          e = p_low*hlf_low*delta_low*n/plf_low # eq. 11.141
          rh = bl*(1.0 - delta_low)*n # eq. 11.142
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL:
          hlf_low = (q_full - bl)/(q_full - q_low) # eq. 11.151
          hlf_full = 1.0 - hlf_low # eq. 11.152
          e = (p_low*hlf_low+p_full*hlf_full)*delta_low*n # eq. 11.150
          rh = bl*(1.0 - delta_low)*n # eq. 11.142
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL:
          hlf_low = (q_full - bl)/(q_full - q_low) # eq. 11.151
          plf_full = 1.0 - self.c_d_heating*(1.0 - hlf_low) # eq. 11.155
          e = p_full*hlf_full*delta_full*n/plf_full # eq. 11.150
          rh = bl*(1.0 - delta_low)*n # eq. 11.142
        else: # elif bl >= q_full
          hlf_full = 1.0 # eq. 11.158
          e = p_full*hlf_full*delta_full*n # eq. 11.156
          rh = (bl - q_full*hlf_full*delta_full)*n # eq. 11.157

      q_sum += n*bl
      e_sum += e
      rh_sum += rh

    t_test = max(self.defrost.period, u(90,"min"))
    t_max  = min(self.defrost.max_time, u(720.0,'min'))

    if self.defrost.control == DefrostControl.DEMAND:
      f_def = 1 + 0.03 * (1 - (t_test-u(90.0,'min'))/(t_max-u(90.0,'min'))) # eq. 11.129
    else:
      f_def = 1 # eq. 11.130

    hspf = q_sum/(e_sum + rh_sum) * f_def # eq. 11.133
    return convert(hspf,'','Btu/Wh')

  def print_cooling_info(self):
    print(f"SEER: {self.seer()}")
    for speed in range(self.number_of_speeds):
      conditions = CoolingConditions(compressor_speed=speed)
      conditions.set_rated_air_flow(self.flow_rated_per_cap_cooling_rated[speed], self.gross_total_cooling_capacity_rated[speed])
      print(f"Net cooling power for stage {speed + 1} : {self.net_cooling_power(conditions)}")
      print(f"Net cooling capacity for stage {speed + 1} : {self.net_total_cooling_capacity(conditions)}")
      print(f"Net SHR for stage {speed + 1} : {self.net_shr(conditions)}")

  def print_heating_info(self, region=4):
    print(f"HSPF (region {region}): {self.hspf(region)}")
    for speed in range(self.number_of_speeds):
      conditions = HeatingConditions(compressor_speed=speed)
      conditions.set_rated_air_flow(self.flow_rated_per_cap_heating_rated[speed], self.gross_heating_capacity_rated[speed])
      print(f"Net heating power for stage {speed + 1} : {self.net_integrated_heating_power(conditions)}")
      print(f"Net heating capacity for stage {speed + 1} : {self.net_integrated_heating_capacity(conditions)}")

  def writeA205(self):
    '''TODO: Write ASHRAE 205 file!!!'''
    return

#%%
# Move this stuff to a separate file

# Single speed
dx_unit_1_speed = DXUnit()

dx_unit_1_speed.print_cooling_info()

dx_unit_1_speed.print_heating_info()

# Two speed
dx_unit_2_speed = DXUnit(
  gross_cooling_cop_rated=[3.0,3.5],
  fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')]*2,
  flow_rated_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  net_total_cooling_capacity_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')],
  fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')]*2,
  gross_heating_cop_rated=[2.5, 3.0],
  flow_rated_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  net_heating_capacity_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')]
)

dx_unit_2_speed.print_cooling_info()

dx_unit_2_speed.print_heating_info()
dx_unit_2_speed.print_heating_info(region=2)

#%%
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
plt.show()
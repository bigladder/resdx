#%%
import sys
from enum import Enum

import numpy as np

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
  return f(cond_1) + (f(cond_2) - f(cond_1))/(cond_2.outdoor_drybulb - cond_1.outdoor_drybulb)*(x - cond_1.outdoor_drybulb)

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
class CoolingConditions:
  def __init__(self,outdoor_drybulb=u(95.0,"°F"),
                    indoor_rh=0.4,
                    indoor_drybulb=u(80.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=1.0, # operating flow / rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction  # TODO: Still need to figure out how to actually use this
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)

class HeatingConditions:
  def __init__(self,outdoor_drybulb=u(47.0,"°F"),
                    indoor_rh=0.4,
                    outdoor_rh=0.4,
                    indoor_drybulb=u(70.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=1.0, # operating flow / rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction  # TODO: Still need to figure out how to actually use this
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)
    self.outdoor_rh = outdoor_rh

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

class DefrostControl(Enum):
  TIMED = 1,
  DEMAND = 2

class DefrostStrategy(Enum):
  REVERSE_CYCLE = 1,
  RESISTIVE = 2

class CyclingMethod(Enum):
  BETWEEN_LOW_FULL = 1
  BETWEEN_OFF_FULL = 2

class DXUnit:

  A_full_cond = CoolingConditions()
  B_full_cond = CoolingConditions(outdoor_drybulb=u(82.0,"°F"))
  B_low_cond = CoolingConditions(outdoor_drybulb=u(82.0,"°F"),compressor_speed=1)
  F_low_cond = CoolingConditions(outdoor_drybulb=u(67.0,"°F"),compressor_speed=1)
  H1_full_cond = HeatingConditions()
  H3_full_cond = HeatingConditions(outdoor_drybulb=u(17.0,"°F"))
  H2_full_cond = HeatingConditions(outdoor_drybulb=u(35.0,"°F"))
  H0_low_cond = HeatingConditions(outdoor_drybulb=u(62.0,"°F"),compressor_speed=1)
  H1_low_cond = HeatingConditions(compressor_speed=1)
  H3_low_cond = HeatingConditions(outdoor_drybulb=u(17.0,"°F"),compressor_speed=1)
  H2_low_cond = HeatingConditions(outdoor_drybulb=u(35.0,"°F"),compressor_speed=1)

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

  def __init__(self,gross_total_cooling_capacity=lambda conditions, scalar : scalar,
                    gross_sensible_cooling_capacity=lambda conditions, scalar : scalar,
                    gross_cooling_power=lambda conditions, scalar : scalar,
                    c_d_cooling=0.2,
                    fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')],
                    cop_cooling_rated=[3.0],
                    flow_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_cooling_rated=[u(3.0,'ton_of_refrigeration')],
                    shr_cooling_rated=[0.8], # Sensible heat ratio (Sensible capacity / Total capacity)
                    gross_stead_state_heating_capacity=lambda conditions, scalar : scalar,
                    gross_integrated_heating_capacity=lambda conditions, scalar1, scalar2, defrost_control, defrost_strategy : scalar1, # scalar1 = timde defrost fraction, scalar2 = heating capacity rated, scalar3 = resistive heater capacity
                    gross_stead_state_heating_power=lambda conditions, scalar : scalar,
                    gross_integrated_heating_power=lambda conditions, scalar1, scalar2, scalar3, defrost_control, defrost_strategy : scalar1,
                    defrost_time_fraction=lambda conditions : u(3.5,'min')/u(60.0,'min'),
                    defrost_resistive_power = 0,
                    c_d_heating=0.2,
                    fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')],
                    cop_heating_rated=[2.5],
                    flow_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_heating_rated=[u(3.0,'ton_of_refrigeration')],
                    defrost_control = DefrostControl.TIMED,
                    defrost_strategy = DefrostStrategy.REVERSE_CYCLE,
                    cycling_method = CyclingMethod.BETWEEN_LOW_FULL,
                    heating_off_temperature = u(10.0,"°F"), # value taken from Scott's script single-stage
                    heating_on_temperature = u(14.0,"°F")): # value taken from Scott's script single-stage

    # Initialize values
    self.number_of_speeds = len(cop_cooling_rated)
    self.gross_total_cooling_capacity = gross_total_cooling_capacity
    self.gross_sensible_cooling_capacity = gross_sensible_cooling_capacity
    self.gross_cooling_power = gross_cooling_power
    self.c_d_cooling = c_d_cooling
    self.fan_eff_cooling_rated = fan_eff_cooling_rated
    self.shr_cooling_rated = shr_cooling_rated
    self.cop_cooling_rated = cop_cooling_rated
    self.cap_cooling_rated = cap_cooling_rated
    self.flow_per_cap_cooling_rated = flow_per_cap_cooling_rated
    self.gross_stead_state_heating_capacity = gross_stead_state_heating_capacity
    self.gross_integrated_heating_capacity = gross_integrated_heating_capacity
    self.gross_stead_state_heating_power = gross_stead_state_heating_power
    self.gross_integrated_heating_power = gross_integrated_heating_power
    self.defrost_time_fraction = defrost_time_fraction
    self.defrost_resistive_power = defrost_resistive_power
    self.c_d_heating = c_d_heating
    self.cycling_method = cycling_method
    self.fan_eff_heating_rated = fan_eff_heating_rated
    self.cop_heating_rated = cop_heating_rated
    self.flow_per_cap_heating_rated = flow_per_cap_heating_rated
    self.cap_heating_rated = cap_heating_rated
    self.defrost_control = defrost_control
    self.defrost_strategy = defrost_strategy
    self.heating_off_temperature = heating_off_temperature
    self.heating_on_temperature = heating_on_temperature

    ## Check for errors

    # Check to make sure all cooling arrays are the same size if not output a warning message
    self.check_array_lengths()

    # Check to make sure arrays are in descending order
    self.check_array_order(self.cap_cooling_rated)
    self.check_array_order(self.cap_heating_rated)

    # TODO: Resistive defrost needs > 0 resistive power

  def check_array_length(self, array):
    if (len(array) != self.number_of_speeds):
      sys.exit(f'Unexpected array length ({len(array)}). Number of speeds is {self.number_of_speeds}. Array items are {array}.')

  def check_array_lengths(self):
    self.check_array_length(self.fan_eff_cooling_rated)
    self.check_array_length(self.flow_per_cap_cooling_rated)
    self.check_array_length(self.cap_cooling_rated)
    self.check_array_length(self.shr_cooling_rated)
    self.check_array_length(self.fan_eff_heating_rated)
    self.check_array_length(self.cop_heating_rated)
    self.check_array_length(self.flow_per_cap_heating_rated)
    self.check_array_length(self.cap_heating_rated)

  def check_array_order(self, array):
    if not all(earlier >= later for earlier, later in zip(array, array[1:])):
      sys.exit(f'Arrays must be in order of decreasing capacity. Array items are {array}.')

  def fan_power(self, conditions):
    if type(conditions) == CoolingConditions:
      return self.fan_eff_cooling_rated[conditions.compressor_speed]*self.standard_indoor_airflow(conditions) # eq. 11.16
    else: # if type(conditions) == HeatingConditions:
      return self.fan_eff_heating_rated[conditions.compressor_speed]*self.standard_indoor_airflow(conditions) # eq. 11.16

  def fan_heat(self, conditions):
    return self.fan_power(conditions) # eq. 11.11 (in SI units)

  def standard_indoor_airflow(self,conditions):
    air_density_standard_conditions = 1.204 # in kg/m3
    if type(conditions) == CoolingConditions:
      flow = self.flow_per_cap_cooling_rated[conditions.compressor_speed]*self.cap_cooling_rated[conditions.compressor_speed]
    else: # if type(conditions) == HeatingConditions:
      flow = self.flow_per_cap_heating_rated[conditions.compressor_speed]*self.cap_heating_rated[conditions.compressor_speed]
    return flow*psychrolib.GetDryAirDensity(convert(conditions.indoor_drybulb,"K","°C"), conditions.press)/air_density_standard_conditions

  ### For cooling ###
  def net_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity(conditions,self.cap_cooling_rated[conditions.compressor_speed]) - self.fan_heat(conditions) # eq. 11.3 but not considering duct losses

  def net_cooling_power(self, conditions):
    return self.gross_cooling_power(conditions,self.cap_cooling_rated[conditions.compressor_speed]/self.cop_cooling_rated[conditions.compressor_speed]) + self.fan_power(conditions) # eq. 11.15

  def eer(self, conditions): # e.q. 11.17
    eer = self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)
    return eer

  def seer(self):
    if self.number_of_speeds == 1:
      plf = 1.0 - 0.5*self.c_d_cooling # eq. 11.56
      seer = plf*self.eer(self.B_full_cond) # eq. 11.55
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
  def net_steady_state_heating_capacity(self, conditions):
    return self.gross_stead_state_heating_capacity(conditions,self.cap_heating_rated[conditions.compressor_speed]) + self.fan_heat(conditions) # eq. 11.31

  def net_steady_state_heating_power(self, conditions):
    return self.gross_stead_state_heating_power(conditions,self.cap_heating_rated[conditions.compressor_speed]/self.cop_heating_rated[conditions.compressor_speed]) - self.fan_power(conditions) # eq. 11.41

  def net_integrated_heating_capacity(self, conditions):
    return self.gross_integrated_heating_capacity(conditions,self.defrost_time_fraction(conditions),self.cap_heating_rated[conditions.compressor_speed], self.defrost_control, self.defrost_strategy) + self.fan_heat(conditions) # eq. 11.31

  def net_integrated_heating_power(self, conditions):
    return self.gross_integrated_heating_power(conditions,self.defrost_time_fraction(conditions),self.cap_heating_rated[conditions.compressor_speed]/self.cop_heating_rated[conditions.compressor_speed],self.defrost_resistive_power, self.defrost_control, self.defrost_strategy) - self.fan_power(conditions) # eq. 11.41

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
        t_ob = u(45,"°F") # eq. 11.134
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

    t_test = u(90.0,'min') # TODO: make input
    t_max  = u(720.0,'min') # TODO: make input

    if self.defrost_control == DefrostControl.DEMAND:
      f_def = 1 + 0.03 * (1 - (t_test-u(90.0,'min'))/(t_max-u(90.0,'min'))) # eq. 11.129
    else:
      f_def = 1 # eq. 11.130

    hspf = q_sum/(e_sum + rh_sum) * f_def # eq. 11.133
    return convert(hspf,'','Btu/Wh')

  def print_cooling_info(self):
    print(f"SEER: {self.seer()}")
    for speed in range(self.number_of_speeds):
      conditions = CoolingConditions(compressor_speed=speed)
      print(f"Net cooling power for stage {speed + 1} : {self.net_cooling_power(conditions)}")
      print(f"Net cooling capacity for stage {speed + 1} : {self.net_total_cooling_capacity(conditions)}")

  def print_heating_info(self, region=4):
    print(f"HSPF (region {region}): {self.hspf(region)}")
    for speed in range(self.number_of_speeds):
      conditions = HeatingConditions(compressor_speed=speed)
      print(f"Net heating power for stage {speed + 1} : {self.net_integrated_heating_power(conditions)}")
      print(f"Net heating capacity for stage {speed + 1} : {self.net_integrated_heating_capacity(conditions)}")

  def writeA205(self):
    '''TODO: Write ASHRAE 205 file!!!'''
    return

def cutler_cooling_power(conditions, scalar):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*scalar

def cutler_total_cooling_capacity(conditions, scalar):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*scalar

def adp_bf_sensible_cooling_capacity(conditions, scalar):
  # TODO: Add function to calculate sensible cooling capacity using the apparatus dew point and bypass factor approach
  return 0.8

def cutler_steady_state_heating_power(conditions, scalar):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*scalar

def cutler_steady_state_heating_capacity(conditions, scalar):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*scalar

def epri_integrated_heating_capacity(conditions, scalar1, scalar2, defrost_control, defrost_strategy):
  # TODO: Do stuff from EPRI report described in EnergyPlus documentation
  if defrost_control ==DefrostControl.TIMED:
      heating_capacity_multiplier = 0.909 - 107.33 * coil_diff_outdoor_air_humidity(conditions)
  else:
      heating_capacity_multiplier = 0.875 * (1-scalar1)

  if defrost_strategy == DefrostStrategy.REVERSE_CYCLE:
      Q_defrost_indoor_u = 0.01 * (7.222 - convert(conditions.outdoor_drybulb,"°K","°C")) * (scalar2/1.01667)
  else:
      Q_defrost_indoor_u = 0

  Q_with_frost_indoor_u = cutler_steady_state_heating_capacity(conditions,scalar2) * heating_capacity_multiplier
  return Q_with_frost_indoor_u * (1-scalar1) - Q_defrost_indoor_u * scalar1 # Do this for now...actual result will be applied on top

def epri_integrated_heating_power(conditions, scalar1, scalar2, scalar3, defrost_control, defrost_strategy):
  # TODO: Do stuff from EPRI report described in EnergyPlus documentation
  if defrost_control == DefrostControl.TIMED:
      input_power_multiplier = 0.9 - 36.45 * coil_diff_outdoor_air_humidity(conditions)
  else:
      input_power_multiplier = 0.954 * (1-scalar1)

  if defrost_strategy == DefrostStrategy.REVERSE_CYCLE:
      P_defrost = 0.1528 * (scalar2/1.01667)
  else:
      P_defrost = scalar3

  P_with_frost = cutler_steady_state_heating_power(conditions,scalar2) * input_power_multiplier
  return P_with_frost * (1-scalar1) + P_defrost * scalar1 # Do this for now...actual result will be applied on top

def epri_defrost_time_fraction(conditions):
  # TODO: Add function for defrost time fraction from EPRI report described in EnergyPlus documentation
  return 1/(1+(0.01446/coil_diff_outdoor_air_humidity(conditions)))

def coil_diff_outdoor_air_humidity(conditions):
  # TODO: Add function for defrost time fraction from EPRI report described in EnergyPlus documentation
  T_coil_outdoor = 0.82 * convert(conditions.outdoor_drybulb,"°K","°C") - 8.589
  outdoor_air_himidity_ratio   = psychrolib.GetHumRatioFromRelHum(convert(conditions.outdoor_drybulb,"°K","°C"),conditions.outdoor_rh,conditions.press)
  saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.press) # pressure in Pa already
  humidity_diff = outdoor_air_himidity_ratio - saturated_air_himidity_ratio
  return max(0.000001,humidity_diff)

#%%
# Move this stuff to a separate file

# Single speed
dx_unit_1_speed = DXUnit(
  gross_total_cooling_capacity=cutler_total_cooling_capacity,
  gross_cooling_power=cutler_cooling_power,
  gross_stead_state_heating_capacity=cutler_steady_state_heating_capacity,
  gross_stead_state_heating_power=cutler_steady_state_heating_power,
  gross_integrated_heating_capacity=epri_integrated_heating_capacity,
  gross_integrated_heating_power=epri_integrated_heating_power,
  defrost_time_fraction=epri_defrost_time_fraction
)

dx_unit_1_speed.print_cooling_info()

dx_unit_1_speed.print_heating_info()

# Two speed
dx_unit_2_speed = DXUnit(
  cop_cooling_rated=[3.0,3.5],
  fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')]*2,
  flow_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  cap_cooling_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')],
  shr_cooling_rated=[0.8]*2,
  gross_total_cooling_capacity=cutler_total_cooling_capacity,
  gross_cooling_power=cutler_cooling_power,
  fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')]*2,
  cop_heating_rated=[2.5, 3.0],
  flow_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  cap_heating_rated=[u(3.0,'ton_of_refrigeration'),u(1.5,'ton_of_refrigeration')],
  gross_stead_state_heating_capacity=cutler_steady_state_heating_capacity,
  gross_stead_state_heating_power=cutler_steady_state_heating_power,
  gross_integrated_heating_capacity=epri_integrated_heating_capacity,
  gross_integrated_heating_power=epri_integrated_heating_power,
  defrost_time_fraction=epri_defrost_time_fraction
)

dx_unit_2_speed.print_cooling_info()

dx_unit_2_speed.print_heating_info()
dx_unit_2_speed.print_heating_info(region=2)

#%%
# Plot integrated power and capacity
P_integrated = []
Q_integrated = []
T_outdoor = []
for T_out in np.arange(-23,75+1,1): #np.arange(-23,40+1,1):
    conditions = HeatingConditions(outdoor_drybulb=u(T_out,"°F"))
    if T_out <= 45:
        Q = epri_integrated_heating_capacity(conditions, dx_unit_1_speed.defrost_time_fraction(conditions), dx_unit_1_speed.cap_heating_rated[0], dx_unit_1_speed.defrost_control, dx_unit_1_speed.defrost_strategy)
        P = epri_integrated_heating_power(conditions, dx_unit_1_speed.defrost_time_fraction(conditions), dx_unit_1_speed.cap_heating_rated[0]/dx_unit_1_speed.cop_heating_rated[0], dx_unit_1_speed.defrost_resistive_power, dx_unit_1_speed.defrost_control, dx_unit_1_speed.defrost_strategy)
    else:
        Q = cutler_steady_state_heating_capacity(conditions, dx_unit_1_speed.cap_heating_rated[0])
        P = cutler_steady_state_heating_power(conditions, dx_unit_1_speed.cap_heating_rated[0]/dx_unit_1_speed.cop_heating_rated[0])
    Q_integrated.append(Q)
    P_integrated.append(P)
    T_outdoor.append(T_out)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Temp (°F)')
ax1.set_ylabel('Capacity (W)', color=color)
ax1.plot(T_outdoor, Q_integrated, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 

color = 'tab:blue'
ax2.set_ylabel('Power (W)', color=color)
ax2.plot(T_outdoor, P_integrated, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
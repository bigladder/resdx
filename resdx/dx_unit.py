import sys
from enum import Enum
import math
from scipy import optimize

from .psychrometrics import PsychState, psychrolib
from .conditions import HeatingConditions, CoolingConditions
from .defrost import Defrost, DefrostControl
from .units import fr_u, to_u
from .util import find_nearest
from .models import RESNETDXModel


def interpolate(f, cond_1, cond_2, x):
  return f(cond_1) + (f(cond_2) - f(cond_1))/(cond_2.outdoor.db - cond_1.outdoor.db)*(x - cond_1.outdoor.db)

class CyclingMethod(Enum):
  BETWEEN_LOW_FULL = 1
  BETWEEN_OFF_FULL = 2

class StagingType(Enum):
  SINGLE_STAGE = 1
  TWO_STAGE = 2
  VARIABLE_SPEED = 3

class AHRIVersion(Enum):
  AHRI_210_240_2017 = 1
  AHRI_210_240_2023 = 2

# AHRI 210/240 2017 distributions
class HeatingDistribution:
  outdoor_drybulbs = [fr_u(62.0 - delta*5.0,"°F") for delta in range(18)] # 62.0 to -23 F by 5 F increments
  def __init__(self,
    outdoor_design_temperature=fr_u(5.0,"°F"),
    fractional_hours=[0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0,0,0],
    c=None,
    c_vs=None,
    zero_load_temperature=None
  ):
    self.outdoor_design_temperature = outdoor_design_temperature
    self.c = c
    self.c_vs = c_vs
    self.zero_load_temperature = zero_load_temperature
    hour_fraction_sum = sum(fractional_hours)
    if hour_fraction_sum < 0.98 or hour_fraction_sum > 1.02:
      # Issue with 2023 standard, unsure how to interpret
      # print(f"Warning: HeatingDistribution sum of fractional hours ({hour_fraction_sum}) is not 1.0.")
      # print(f"         Values will be re-normalized.")
      self.fractional_hours = [n/hour_fraction_sum for n in fractional_hours]
    else:
      self.fractional_hours = fractional_hours
    self.number_of_bins = len(self.fractional_hours)
    if self.number_of_bins != 18:
      raise Exception(f'Heating distributions must be provided in 18 bins.')


class CoolingDistribution:
  outdoor_drybulbs = [fr_u(67.0 + delta*5.0,"°F") for delta in range(8)] # 67.0 to 102 F by 5 F increments
  fractional_hours = [0.214,0.231,0.216,0.161,0.104,0.052,0.018,0.004]

  def __init__(self):
    self.number_of_bins = len(self.fractional_hours)

class DXUnit:

  regional_heating_distributions = {
    AHRIVersion.AHRI_210_240_2017: {
      1: HeatingDistribution(fr_u(37.0,"°F"), [0.291,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
      2: HeatingDistribution(fr_u(27.0,"°F"), [0.215,0.189,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]),
      3: HeatingDistribution(fr_u(17.0,"°F"), [0.153,0.142,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000]),
      4: HeatingDistribution(fr_u(5.0,"°F"),  [0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0.000,0.000,0.000]),
      5: HeatingDistribution(fr_u(-10.0,"°F"),[0.106,0.092,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001]),
      6: HeatingDistribution(fr_u(30.0,"°F"), [0.113,0.206,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000])},
    AHRIVersion.AHRI_210_240_2023: {
      # Note: AHRI 2023 issue: None of these distributions add to 1.0!
      1: HeatingDistribution(fr_u(37.0,"°F"), [0.000,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.10, 1.03, fr_u(58.0,"°F")),
      2: HeatingDistribution(fr_u(27.0,"°F"), [0.000,0.000,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.06, 0.99, fr_u(57.0,"°F")),
      3: HeatingDistribution(fr_u(17.0,"°F"), [0.000,0.000,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0.000,0.000,0.000,0.000,0.000], 1.30, 1.21, fr_u(56.0,"°F")),
      4: HeatingDistribution(fr_u(5.0,"°F"),  [0.000,0.000,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0.000,0.000,0.000], 1.15, 1.07, fr_u(55.0,"°F")),
      5: HeatingDistribution(fr_u(-10.0,"°F"),[0.000,0.000,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001], 1.16, 1.08, fr_u(55.0,"°F")),
      6: HeatingDistribution(fr_u(30.0,"°F"), [0.000,0.000,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000], 1.11, 1.03, fr_u(57.0,"°F"))},
  }

  cooling_distribution = CoolingDistribution()

  standard_design_heating_requirements = [(fr_u(5000,"Btu/hr")+i*fr_u(5000,"Btu/hr")) for i in range(0,8)] + [(fr_u(50000,"Btu/hr")+i*fr_u(10000,"Btu/hr")) for i in range(0,9)]

  def __init__(self,model = RESNETDXModel(),
                    # defaults of None are defaulted within this function based on other argument values
                    number_of_input_stages=None,
                    net_total_cooling_capacity_rated = fr_u(3.0,'ton_of_refrigeration'),
                    gross_cooling_cop_rated = 3.72,
                    net_cooling_cop_rated = None,
                    fan_efficacy_cooling_rated = None,
                    flow_rated_per_cap_cooling_rated = None, # Per net total cooling capacity
                    c_d_cooling = None,
                    net_heating_capacity_rated = None,
                    gross_heating_cop_rated = 3.82,
                    net_heating_cop_rated = None,
                    fan_efficacy_heating_rated = None,
                    flow_rated_per_cap_heating_rated = None, # Per net total cooling capacity,
                    c_d_heating = None,
                    heating_off_temperature = fr_u(0.0,"°F"),
                    heating_on_temperature = None, # default to heating_off_temperature
                    defrost = Defrost(),
                    cycling_method = CyclingMethod.BETWEEN_LOW_FULL,
                    full_load_speed = 0, # The first entry (index = 0) in arrays reflects AHRI "full" speed.
                    intermediate_speed = None,
                    staging_type = None, # Allow default based on inputs
                    rating_standard = AHRIVersion.AHRI_210_240_2017,

                    # Used for comparisons and to inform some defaults
                    input_seer = None,
                    input_hspf = None,
                    **kwargs):  # Additional inputs used for specific models

    # Initialize direct values
    self.kwargs = kwargs

    self.model = model
    self.model.set_system(self)

    self.input_seer = input_seer
    self.input_hspf = input_hspf
    self.cycling_method = cycling_method
    self.defrost = defrost
    self.rating_standard = rating_standard
    self.heating_off_temperature = heating_off_temperature
    if heating_on_temperature == None:
      self.heating_on_temperature = self.heating_off_temperature

    # Placeholder for additional data set specific to the model
    self.model_data = {}

    # Number of stages/speeds
    if number_of_input_stages is not None:
      self.number_of_input_stages = number_of_input_stages
      if type(net_total_cooling_capacity_rated) is list:
        num_capacities = len(net_total_cooling_capacity_rated)
        if num_capacities != number_of_input_stages:
          raise Exception(f'Length of \'net_total_cooling_capacity_rated\' ({num_capacities}) != \'number_of_input_stages\' ({number_of_input_stages}).')
    elif type(net_total_cooling_capacity_rated) is list:
      self.number_of_input_stages = len(net_total_cooling_capacity_rated)
    else:
      self.number_of_input_stages = 1

    self.full_load_speed = full_load_speed

    if intermediate_speed is None:
      self.intermediate_speed = full_load_speed + 1
    else:
      self.intermediate_speed = intermediate_speed

    self.low_speed = self.number_of_input_stages - 1

    if staging_type is None:
      self.staging_type = StagingType(min(self.number_of_input_stages,3))
    else:
      self.staging_type = staging_type

    # Placeholders for derived staging array values
    self.set_placeholder_arrays()

    # Default staging array inputs
    self.model.set_fan_efficacy_cooling_rated(fan_efficacy_cooling_rated)
    self.model.set_fan_efficacy_heating_rated(fan_efficacy_heating_rated)
    self.model.set_flow_rated_per_cap_cooling_rated(flow_rated_per_cap_cooling_rated)
    self.model.set_flow_rated_per_cap_heating_rated(flow_rated_per_cap_heating_rated)

    # Set net capacities in case lower speed values depend on gross capacities
    self.model.set_net_total_cooling_capacity_rated(net_total_cooling_capacity_rated)
    self.model.set_net_heating_capacity_rated(net_heating_capacity_rated)

    # Degradation coefficients
    self.model.set_c_d_cooling(c_d_cooling)
    self.model.set_c_d_heating(c_d_heating)

    # Derived staging array values
    for i in range(self.number_of_input_stages):
      self.cooling_fan_power_rated[i] = self.net_total_cooling_capacity_rated[i]*self.fan_efficacy_cooling_rated[i]*self.flow_rated_per_cap_cooling_rated[i]
      self.heating_fan_power_rated[i] = self.net_total_cooling_capacity_rated[i]*self.fan_efficacy_heating_rated[i]*self.flow_rated_per_cap_heating_rated[i] # note: heating fan flow is intentionally based on cooling capacity
      self.gross_total_cooling_capacity_rated[i] = self.net_total_cooling_capacity_rated[i] + self.cooling_fan_power_rated[i]
      self.gross_heating_capacity_rated[i] = self.net_heating_capacity_rated[i] - self.heating_fan_power_rated[i]

    # COP determinations
    if net_cooling_cop_rated is None and gross_cooling_cop_rated is None:
      raise Exception(f'Must define either \'net_cooling_cop_rated\' or \'gross_cooling_cop_rated\'.')

    if net_heating_cop_rated is None and gross_heating_cop_rated is None:
      raise Exception(f'Must define either \'net_heating_cop_rated\' or \'gross_heating_cop_rated\'.')

    if net_cooling_cop_rated is not None:
      self.model.set_net_cooling_cop_rated(net_cooling_cop_rated)

    if gross_cooling_cop_rated is not None:
      self.model.set_gross_cooling_cop_rated(gross_cooling_cop_rated)

    if net_heating_cop_rated is not None:
      self.model.set_net_heating_cop_rated(net_heating_cop_rated)

    if gross_heating_cop_rated is not None:
      self.model.set_gross_heating_cop_rated(gross_heating_cop_rated)

    ## Set rating conditions TODO: re-set when AHRIStandard version changes (for ESP)
    self.A_full_cond = self.make_condition(CoolingConditions,compressor_speed=self.full_load_speed)
    self.B_full_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=fr_u(82.0,"°F"),wetbulb=fr_u(65.0,"°F")),compressor_speed=self.full_load_speed)

    self.H1_full_cond = self.make_condition(HeatingConditions,compressor_speed=self.full_load_speed)
    self.H2_full_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(35.0,"°F"),wetbulb=fr_u(33.0,"°F")),compressor_speed=self.full_load_speed)
    self.H3_full_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(17.0,"°F"),wetbulb=fr_u(15.0,"°F")),compressor_speed=self.full_load_speed)
    self.H4_full_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(5.0,"°F"),wetbulb=fr_u(3.0,"°F")),compressor_speed=self.full_load_speed)

    self.gross_shr_cooling_rated = [self.model.gross_shr(self.A_full_cond)]
    self.calculate_bypass_factor_rated(self.A_full_cond)

    if self.staging_type != StagingType.SINGLE_STAGE:
      if self.staging_type == StagingType.VARIABLE_SPEED:
        self.A_int_cond = self.make_condition(CoolingConditions,compressor_speed=self.intermediate_speed) # Not used in AHRI ratings, only used for 'rated' SHR calculations at low speeds
        self.E_int_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=fr_u(87.0,"°F"),wetbulb=fr_u(69.0,"°F")),compressor_speed=self.intermediate_speed)

        self.H2_int_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(35.0,"°F"),wetbulb=fr_u(33.0,"°F")),compressor_speed=self.intermediate_speed)

        self.gross_shr_cooling_rated += [self.model.gross_shr(self.A_int_cond)]
        self.calculate_bypass_factor_rated(self.A_int_cond)

      self.A_low_cond = self.make_condition(CoolingConditions,compressor_speed=self.low_speed) # Not used in AHRI ratings, only used for 'rated' SHR calculations at low speeds
      self.B_low_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=fr_u(82.0,"°F"),wetbulb=fr_u(65.0,"°F")),compressor_speed=self.low_speed)
      self.F_low_cond = self.make_condition(CoolingConditions,outdoor=PsychState(drybulb=fr_u(67.0,"°F"),wetbulb=fr_u(53.5,"°F")),compressor_speed=self.low_speed)

      self.H0_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(62.0,"°F"),wetbulb=fr_u(56.5,"°F")),compressor_speed=self.low_speed)
      self.H1_low_cond = self.make_condition(HeatingConditions,compressor_speed=1)
      self.H2_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(35.0,"°F"),wetbulb=fr_u(33.0,"°F")),compressor_speed=self.low_speed)
      self.H3_low_cond = self.make_condition(HeatingConditions,outdoor=PsychState(drybulb=fr_u(17.0,"°F"),wetbulb=fr_u(15.0,"°F")),compressor_speed=self.low_speed)

      self.gross_shr_cooling_rated += [self.model.gross_shr(self.A_low_cond)]
      self.calculate_bypass_factor_rated(self.A_low_cond)

    ## Check for errors

    # Check to make sure all cooling arrays are the same size if not output a warning message
    self.check_array_lengths()

    # Check to make sure arrays are in descending order
    self.check_array_order(self.net_total_cooling_capacity_rated)
    self.check_array_order(self.net_heating_capacity_rated)

  def check_array_length(self, array):
    if (len(array) != self.number_of_input_stages):
      raise Exception(f'Unexpected array length ({len(array)}). Number of speeds is {self.number_of_input_stages}. Array items are {array}.')

  def check_array_lengths(self):
    self.check_array_length(self.fan_efficacy_cooling_rated)
    self.check_array_length(self.flow_rated_per_cap_cooling_rated)
    self.check_array_length(self.net_total_cooling_capacity_rated)
    self.check_array_length(self.gross_shr_cooling_rated)
    self.check_array_length(self.fan_efficacy_heating_rated)
    self.check_array_length(self.gross_heating_cop_rated)
    self.check_array_length(self.flow_rated_per_cap_heating_rated)
    self.check_array_length(self.net_heating_capacity_rated)

  def check_array_order(self, array):
    if not all(earlier >= later for earlier, later in zip(array, array[1:])):
      raise Exception(f'Arrays must be in order of decreasing capacity. Array items are {array}.')

  def set_placeholder_arrays(self):
    self.cooling_fan_power_rated = [None]*self.number_of_input_stages
    self.heating_fan_power_rated = [None]*self.number_of_input_stages
    self.gross_total_cooling_capacity_rated = [None]*self.number_of_input_stages
    self.gross_heating_capacity_rated = [None]*self.number_of_input_stages
    self.bypass_factor_rated = [None]*self.number_of_input_stages
    self.normalized_ntu = [None]*self.number_of_input_stages

    self.net_cooling_cop_rated = [None]*self.number_of_input_stages
    self.net_cooling_power_rated = [None]*self.number_of_input_stages
    self.gross_cooling_power_rated = [None]*self.number_of_input_stages
    self.gross_cooling_cop_rated = [None]*self.number_of_input_stages

    self.net_heating_cop_rated = [None]*self.number_of_input_stages
    self.net_heating_power_rated = [None]*self.number_of_input_stages
    self.gross_heating_power_rated = [None]*self.number_of_input_stages
    self.gross_heating_cop_rated = [None]*self.number_of_input_stages

  def make_condition(self, condition_type, compressor_speed=0, indoor=None, outdoor=None):
    if indoor is None:
      indoor = condition_type().indoor
    if outdoor is None:
      outdoor = condition_type().outdoor

    condition = condition_type(indoor=indoor, outdoor=outdoor, compressor_speed=compressor_speed)
    if condition_type == CoolingConditions:
      condition.set_rated_air_flow(self.flow_rated_per_cap_cooling_rated[compressor_speed], self.net_total_cooling_capacity_rated[compressor_speed])
    else: # if condition_type == HeatingConditions:
      condition.set_rated_air_flow(self.flow_rated_per_cap_heating_rated[compressor_speed], self.net_total_cooling_capacity_rated[compressor_speed])
    return condition

  ### For cooling ###
  def cooling_fan_power(self, conditions=None):
    if conditions is None:
      conditions = self.A_full_cond
    # TODO: Change to calculated fan efficacy under current conditions (e.g., at realistic static pressure)
    return self.fan_efficacy_cooling_rated[conditions.compressor_speed]*conditions.std_air_vol_flow

  def cooling_fan_heat(self, conditions):
    return self.cooling_fan_power(conditions)

  def gross_total_cooling_capacity(self, conditions=None):
    if conditions is None:
      conditions = self.A_full_cond
    return self.model.gross_total_cooling_capacity(conditions)

  def gross_sensible_cooling_capacity(self, conditions=None):
    if conditions is None:
      conditions = self.A_full_cond
    return self.model.gross_sensible_cooling_capacity(conditions)

  def gross_shr(self, conditions=None):
    return self.gross_sensible_cooling_capacity(conditions)/self.gross_total_cooling_capacity(conditions)

  def gross_cooling_power(self, conditions=None):
    if conditions is None:
      conditions = self.A_full_cond
    return self.model.gross_cooling_power(conditions)

  def net_total_cooling_capacity(self, conditions=None):
    return self.gross_total_cooling_capacity(conditions) - self.cooling_fan_heat(conditions)

  def net_sensible_cooling_capacity(self, conditions=None):
    return self.gross_sensible_cooling_capacity(conditions) - self.cooling_fan_heat(conditions)

  def net_shr(self, conditions=None):
    return self.net_sensible_cooling_capacity(conditions)/self.net_total_cooling_capacity(conditions)

  def net_cooling_power(self, conditions=None):
    return self.gross_cooling_power(conditions) + self.cooling_fan_power(conditions)

  def gross_cooling_cop(self, conditions=None):
    return self.gross_total_cooling_capacity(conditions)/self.gross_cooling_power(conditions)

  def net_cooling_cop(self, conditions=None):
    return self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)

  def gross_cooling_outlet_state(self, conditions=None, gross_sensible_capacity=None):
    if conditions is None:
      conditions = self.A_full_cond
    if gross_sensible_capacity is None:
      gross_sensible_capacity = self.gross_sensible_cooling_capacity(conditions)

    T_idb = conditions.indoor.db
    h_i = conditions.indoor.get_h()
    m_dot_rated = conditions.air_mass_flow
    h_o = h_i - self.gross_total_cooling_capacity(conditions)/m_dot_rated
    T_odb = T_idb - gross_sensible_capacity/(m_dot_rated*conditions.indoor.C_p)
    return PsychState(T_odb,pressure=conditions.indoor.p,enthalpy=h_o)

  def calculate_adp_state(self, inlet_state, outlet_state):
    T_idb = inlet_state.db_C
    w_i = inlet_state.get_hr()
    T_odb = outlet_state.db_C
    w_o = outlet_state.get_hr()
    root_fn = lambda T_ADP : psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, inlet_state.p) - (w_i - (w_i - w_o)/(T_idb - T_odb)*(T_idb - T_ADP))
    T_ADP = optimize.newton(root_fn, T_idb)
    w_ADP = w_i - (w_i - w_o)/(T_idb - T_odb)*(T_idb - T_ADP)
    # Output an error if ADP calculation method is not applicable:
    if (T_odb < T_ADP or w_o <  w_ADP):
      raise Exception(f'Invalid Apparatus Dew Point (ADP). The rated Sensible Heat Ratio (SHR) might not be valid.')
    return PsychState(fr_u(T_ADP,"°C"),pressure=inlet_state.p,hum_rat=w_ADP)

  def calculate_bypass_factor_rated(self, conditions): # for rated flow rate
    Q_s_rated = self.gross_shr_cooling_rated[conditions.compressor_speed]*self.gross_total_cooling_capacity(conditions)
    outlet_state = self.gross_cooling_outlet_state(conditions,gross_sensible_capacity=Q_s_rated)
    ADP_state = self.calculate_adp_state(conditions.indoor,outlet_state)
    h_i = conditions.indoor.get_h()
    h_o = outlet_state.get_h()
    h_ADP = ADP_state.get_h()
    self.bypass_factor_rated[conditions.compressor_speed] = (h_o - h_ADP)/(h_i - h_ADP)
    self.normalized_ntu[conditions.compressor_speed] = - conditions.air_mass_flow * math.log(self.bypass_factor_rated[conditions.compressor_speed]) # A0 = - m_dot * ln(BF)

  def bypass_factor(self,conditions=None):
    if conditions is None:
      conditions = self.A_full_cond
    return math.exp(-self.normalized_ntu[conditions.compressor_speed]/conditions.air_mass_flow)

  def adp_state(self, conditions=None):
    outlet_state = self.gross_cooling_outlet_state(conditions)
    return self.calculate_adp_state(conditions.indoor,outlet_state)

  def eer(self, conditions=None):
    return to_u(self.net_cooling_cop(conditions),'Btu/Wh')

  def seer(self):
    '''Based on AHRI 210/240 2023 (unless otherwise noted)'''
    if self.staging_type == StagingType.SINGLE_STAGE:
      plf = 1.0 - 0.5*self.c_d_cooling # eq. 11.56
      seer = plf*self.net_cooling_cop(self.B_full_cond) # eq. 11.55 (using COP to keep things in SI units for now)
    else: #if self.staging_type == StagingType.TWO_STAGE or self.staging_type == StagingType.VARIABLE_SPEED:
      sizing_factor = 1.1 # eq. 11.61
      q_sum = 0.0
      e_sum = 0.0
      if self.staging_type == StagingType.VARIABLE_SPEED:
        # Intermediate capacity
        q_A_full = self.net_total_cooling_capacity(self.A_full_cond)
        q_B_full = self.net_total_cooling_capacity(self.B_full_cond)
        q_B_low = self.net_total_cooling_capacity(self.B_low_cond)
        q_F_low = self.net_total_cooling_capacity(self.F_low_cond)
        q_E_int = self.net_total_cooling_capacity(self.E_int_cond)
        q_87_low = interpolate(self.net_total_cooling_capacity, self.F_low_cond, self.B_low_cond, fr_u(87.0,"°F"))
        q_87_full = interpolate(self.net_total_cooling_capacity, self.B_full_cond, self.A_full_cond, fr_u(87.0,"°F"))
        N_Cq = (q_E_int - q_87_low)/(q_87_full - q_87_low)
        M_Cq = (q_B_low - q_F_low)/(fr_u(82,"°F") - fr_u(67.0,"°F"))*(1. - N_Cq) + (q_A_full - q_B_full)/(fr_u(95,"°F") - fr_u(82.0,"°F"))*N_Cq

        # Intermediate power
        p_A_full = self.net_cooling_power(self.A_full_cond)
        p_B_full = self.net_cooling_power(self.B_full_cond)
        p_B_low = self.net_cooling_power(self.B_low_cond)
        p_F_low = self.net_cooling_power(self.F_low_cond)
        p_E_int = self.net_cooling_power(self.E_int_cond)
        p_87_low = interpolate(self.net_cooling_power, self.F_low_cond, self.B_low_cond, fr_u(87.0,"°F"))
        p_87_full = interpolate(self.net_cooling_power, self.B_full_cond, self.A_full_cond, fr_u(87.0,"°F"))
        N_CE = (p_E_int - p_87_low)/(p_87_full - p_87_low)
        M_CE = (p_B_low - p_F_low)/(fr_u(82,"°F") - fr_u(67.0,"°F"))*(1. - N_CE) + (p_A_full - p_B_full)/(fr_u(95,"°F") - fr_u(82.0,"°F"))*N_CE

      for i in range(self.cooling_distribution.number_of_bins):
        t = self.cooling_distribution.outdoor_drybulbs[i]
        n = self.cooling_distribution.fractional_hours[i]
        bl = (t - fr_u(65.0,"°F"))/(fr_u(95,"°F") - fr_u(65.0,"°F"))*self.net_total_cooling_capacity(self.A_full_cond)/sizing_factor # eq. 11.60
        q_low = interpolate(self.net_total_cooling_capacity, self.F_low_cond, self.B_low_cond, t) # eq. 11.62
        p_low = interpolate(self.net_cooling_power, self.F_low_cond, self.B_low_cond, t) # eq. 11.63
        q_full = interpolate(self.net_total_cooling_capacity, self.B_full_cond, self.A_full_cond, t) # eq. 11.64
        p_full = interpolate(self.net_cooling_power, self.B_full_cond, self.A_full_cond, t) # eq. 11.65
        if self.staging_type == StagingType.TWO_STAGE:
          if bl <= q_low:
            clf_low = bl/q_low # eq. 11.68
            plf_low = 1.0 - self.c_d_cooling*(1.0 - clf_low) # eq. 11.69
            q = clf_low*q_low*n # eq. 11.66
            e = clf_low*p_low*n/plf_low # eq. 11.67
          elif bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL:
            clf_low = (q_full - bl)/(q_full - q_low) # eq. 11.74
            clf_full = 1.0 - clf_low # eq. 11.75
            q = (clf_low*q_low + clf_full*q_full)*n # eq. 11.72
            e = (clf_low*p_low + clf_full*p_full)*n # eq. 11.73
          elif bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL:
            clf_full = bl/q_full # eq. 11.78
            plf_full = 1.0 - self.c_d_cooling*(1.0 - clf_full) # eq. 11.79
            q = clf_full*q_full*n # eq. 11.76
            e = clf_full*p_full*n/plf_full # eq. 11.77
          else: # elif bl >= q_full
            q = q_full*n
            e = p_full*n
        else:  # if self.staging_type == StagingType.VARIABLE_SPEED
          q_int = q_E_int + M_Cq*(t - (fr_u(87,"°F")))
          p_int = p_E_int + M_CE*(t - (fr_u(87,"°F")))
          cop_low = q_low/p_low
          cop_int = q_int/p_int
          cop_full = q_full/p_full

          if bl <= q_low:
            clf_low = bl/q_low # eq. 11.68
            plf_low = 1.0 - self.c_d_cooling*(1.0 - clf_low) # eq. 11.69
            q = clf_low*q_low*n # eq. 11.66
            e = clf_low*p_low*n/plf_low # eq. 11.67
          elif bl < q_int:
            cop_int_bin = cop_low + (cop_int - cop_low)/(q_int - q_low)*(bl - q_low) # eq. 11.101 (2023)
            q = bl*n
            e = q/cop_int_bin
          elif bl <= q_full:
            cop_int_bin = cop_int + (cop_full - cop_int)/(q_full - q_int)*(bl - q_int) # eq. 11.101 (2023)
            q = bl*n
            e = q/cop_int_bin
          else: # elif bl >= q_full
            q = q_full*n
            e = p_full*n

        q_sum += q
        e_sum += e

      seer = q_sum/e_sum # e.q. 11.59
    return to_u(seer,'Btu/Wh')

  ### For heating ###
  def heating_fan_power(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    # TODO: Change to calculated fan efficacy under current conditions (e.g., at realistic static pressure)
    return self.fan_efficacy_heating_rated[conditions.compressor_speed]*conditions.std_air_vol_flow

  def heating_fan_heat(self, conditions):
    return self.heating_fan_power(conditions)

  def gross_steady_state_heating_capacity(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    return self.model.gross_steady_state_heating_capacity(conditions)

  def gross_steady_state_heating_power(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    return self.model.gross_steady_state_heating_power(conditions)

  def gross_integrated_heating_capacity(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    return self.model.gross_integrated_heating_capacity(conditions)

  def gross_integrated_heating_power(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    return self.model.gross_integrated_heating_power(conditions)

  def net_steady_state_heating_capacity(self, conditions=None):
    return self.gross_steady_state_heating_capacity(conditions) + self.heating_fan_heat(conditions)

  def net_steady_state_heating_power(self, conditions=None):
    return self.gross_steady_state_heating_power(conditions) + self.heating_fan_power(conditions)

  def net_integrated_heating_capacity(self, conditions=None):
    return self.gross_integrated_heating_capacity(conditions) + self.heating_fan_heat(conditions)

  def net_integrated_heating_power(self, conditions=None):
    return self.gross_integrated_heating_power(conditions) + self.heating_fan_power(conditions)

  def gross_steady_state_heating_cop(self, conditions=None):
    return self.gross_steady_state_heating_capacity(conditions)/self.gross_steady_state_heating_power(conditions)

  def gross_integrated_heating_cop(self, conditions=None):
    return self.gross_integrated_heating_capacity(conditions)/self.gross_integrated_heating_power(conditions)

  def net_steady_state_heating_cop(self, conditions=None):
    return self.net_steady_state_heating_capacity(conditions)/self.net_steady_state_heating_power(conditions)

  def net_integrated_heating_cop(self, conditions=None):
    return self.net_integrated_heating_capacity(conditions)/self.net_integrated_heating_power(conditions)

  def gross_heating_capacity_ratio(self, conditions=None):
    return self.gross_integrated_heating_capacity(conditions)/self.gross_steady_state_heating_capacity(conditions)

  def net_heating_capacity_ratio(self, conditions=None):
    return self.net_integrated_heating_capacity(conditions)/self.net_steady_state_heating_capacity(conditions)

  def gross_heating_power_ratio(self, conditions=None):
    return self.gross_integrated_heating_power(conditions)/self.gross_steady_state_heating_power(conditions)

  def net_heating_power_ratio(self, conditions=None):
    return self.net_integrated_heating_power(conditions)/self.net_steady_state_heating_power(conditions)

  def gross_heating_output_state(self, conditions=None):
    if conditions is None:
      conditions = self.H1_full_cond
    T_odb = conditions.indoor.db + self.gross_steady_state_heating_capacity(conditions)/(conditions.air_mass_flow*conditions.indoor.C_p)
    return PsychState(T_odb,pressure=conditions.indoor.p,hum_rat=conditions.indoor.get_hr())

  def hspf(self, region=4):
    '''Based on AHRI 210/240 2023 (unless otherwise noted)'''
    q_sum = 0.0
    e_sum = 0.0
    rh_sum = 0.0

    heating_distribution = self.regional_heating_distributions[self.rating_standard][region]
    t_od = heating_distribution.outdoor_design_temperature

    if self.rating_standard == AHRIVersion.AHRI_210_240_2017:
      c = 0.77 # eq. 11.110 (agreement factor)
      dhr_min = self.net_integrated_heating_capacity(self.H1_full_cond)*(fr_u(65,"°F")-t_od)/(fr_u(60,"°R")) # eq. 11.111
      dhr_min = find_nearest(self.standard_design_heating_requirements, dhr_min)
    else: # if self.rating_standard == AHRIVersion.AHRI_210_240_2023:
      if self.staging_type == StagingType.VARIABLE_SPEED:
        c_x = heating_distribution.c_vs
      else:
        c_x = heating_distribution.c
      t_zl = heating_distribution.zero_load_temperature


    if self.staging_type == StagingType.VARIABLE_SPEED:
      # Intermediate capacity
      q_H0_low = self.net_integrated_heating_capacity(self.H0_low_cond)
      q_H1_low = self.net_integrated_heating_capacity(self.H1_low_cond)
      q_H2_int = self.net_integrated_heating_capacity(self.H2_int_cond)
      q_H1_full = self.net_integrated_heating_capacity(self.H1_full_cond)
      q_H2_full = self.net_integrated_heating_capacity(self.H2_full_cond)
      q_H3_full = self.net_integrated_heating_capacity(self.H3_full_cond)
      q_H4_full = self.net_integrated_heating_capacity(self.H4_full_cond)
      q_35_low = interpolate(self.net_integrated_heating_capacity, self.H0_low_cond, self.H1_low_cond, fr_u(35.0,"°F"))
      N_Hq = (q_H2_int - q_35_low)/(q_H2_full - q_35_low)
      M_Hq = (q_H0_low - q_H1_low)/(fr_u(62,"°F") - fr_u(47.0,"°F"))*(1. - N_Hq) + (q_H2_full - q_H3_full)/(fr_u(35,"°F") - fr_u(17.0,"°F"))*N_Hq

      # Intermediate power
      p_H0_low = self.net_integrated_heating_power(self.H0_low_cond)
      p_H1_low = self.net_integrated_heating_power(self.H1_low_cond)
      p_H2_int = self.net_integrated_heating_power(self.H2_int_cond)
      p_H1_full = self.net_integrated_heating_power(self.H1_full_cond)
      p_H2_full = self.net_integrated_heating_power(self.H2_full_cond)
      p_H3_full = self.net_integrated_heating_power(self.H3_full_cond)
      p_H4_full = self.net_integrated_heating_power(self.H4_full_cond)
      p_35_low = interpolate(self.net_integrated_heating_power, self.H0_low_cond, self.H1_low_cond, fr_u(35.0,"°F"))
      N_HE = (p_H2_int - p_35_low)/(p_H2_full - p_35_low)
      M_HE = (p_H0_low - p_H1_low)/(fr_u(62,"°F") - fr_u(47.0,"°F"))*(1. - N_Hq) + (p_H2_full - p_H3_full)/(fr_u(35,"°F") - fr_u(17.0,"°F"))*N_Hq

    for i in range(heating_distribution.number_of_bins):
      t = heating_distribution.outdoor_drybulbs[i]
      n = heating_distribution.fractional_hours[i]
      if self.rating_standard == AHRIVersion.AHRI_210_240_2017:
        bl = (fr_u(65,"°F")-t)/(fr_u(65,"°F")-t_od)*c*dhr_min # eq. 11.109
      else: # if self.rating_standard == AHRIVersion.AHRI_210_240_2023:
        q_A_full = self.net_total_cooling_capacity(self.A_full_cond)
        bl = (t_zl-t)/(t_zl-t_od)*c_x*q_A_full

      t_ob = fr_u(45,"°F") # eq. 11.119
      if t >= t_ob or t <= fr_u(17,"°F"):
        q_full = interpolate(self.net_integrated_heating_capacity, self.H3_full_cond, self.H1_full_cond, t) # eq. 11.117
        p_full = interpolate(self.net_integrated_heating_power, self.H3_full_cond, self.H1_full_cond, t) # eq. 11.117
      else: # elif t > fr_u(17,"°F") and t < t_ob
        q_full = interpolate(self.net_integrated_heating_capacity, self.H3_full_cond, self.H2_full_cond, t) # eq. 11.118
        p_full = interpolate(self.net_integrated_heating_power, self.H3_full_cond, self.H2_full_cond, t) # eq. 11.117
      cop_full = q_full/p_full

      if t <= self.heating_off_temperature or cop_full < 1.0:
        delta_full = 0.0 # eq. 11.125
      elif t > self.heating_on_temperature:
        delta_full = 1.0 # eq. 11.127
      else:
        delta_full = 0.5 # eq. 11.126

      if q_full > bl:
        hlf_full = bl/q_full # eq. 11.115 & 11.154
      else:
        hlf_full = 1.0 # eq. 11.116

      if self.staging_type == StagingType.SINGLE_STAGE:
        plf_full = 1.0 - self.c_d_heating*(1.0 - hlf_full) # eq. 11.125
        e = p_full*hlf_full*delta_full*n/plf_full # eq. 11.156 (not shown for single stage)
        rh = (bl - q_full*hlf_full*delta_full)*n # eq. 11.126
      elif self.staging_type == StagingType.TWO_STAGE:
        t_ob = fr_u(40,"°F") # eq. 11.134
        if t >= t_ob:
          q_low = interpolate(self.net_integrated_heating_capacity, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.135
          p_low = interpolate(self.net_integrated_heating_power, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.138
        elif t <= fr_u(17.0,"°F"):
          q_low = interpolate(self.net_integrated_heating_capacity, self.H1_low_cond, self.H3_low_cond, t) # eq. 11.137
          p_low = interpolate(self.net_integrated_heating_power, self.H1_low_cond, self.H3_low_cond, t) # eq. 11.140
        else:
          q_low = interpolate(self.net_integrated_heating_capacity, self.H2_low_cond, self.H3_low_cond, t) # eq. 11.136
          p_low = interpolate(self.net_integrated_heating_power, self.H2_low_cond, self.H3_low_cond, t) # eq. 11.139

        cop_low = q_low/p_low
        if bl <= q_low:
          if t <= self.heating_off_temperature or cop_low < 1.0:
            delta_low = 0.0 # eq. 11.159
          elif t > self.heating_on_temperature:
            delta_low = 1.0 # eq. 11.160
          else:
            delta_low = 0.5 # eq. 11.161

          hlf_low = bl/q_low # eq. 11.155
          plf_low = 1.0 - self.c_d_heating*(1.0 - hlf_low) # eq. 11.156
          e = p_low*hlf_low*delta_low*n/plf_low # eq. 11.153
          rh = bl*(1.0 - delta_low)*n # eq. 11.154
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_LOW_FULL:
          hlf_low = (q_full - bl)/(q_full - q_low) # eq. 11.163
          hlf_full = 1.0 - hlf_low # eq. 11.164
          e = (p_low*hlf_low+p_full*hlf_full)*delta_low*n # eq. 11.162
          rh = bl*(1.0 - delta_low)*n # eq. 11.154
        elif bl > q_low and bl < q_full and self.cycling_method == CyclingMethod.BETWEEN_OFF_FULL:
          hlf_low = (q_full - bl)/(q_full - q_low) # eq. 11.163
          plf_full = 1.0 - self.c_d_heating*(1.0 - hlf_low) # eq. 11.166
          e = p_full*hlf_full*delta_full*n/plf_full # eq. 11.165
          rh = bl*(1.0 - delta_low)*n # eq. 11.142
        else: # elif bl >= q_full
          hlf_full = 1.0 # eq. 11.170
          e = p_full*hlf_full*delta_full*n # eq. 11.168
          rh = (bl - q_full*hlf_full*delta_full)*n # eq. 11.169
      else: # if self.staging_type == StagingType.VARIABLE_SPEED:
        # Note: this is strange that there is no defrost cut in the low speed and doesn't use H2 or H3 low
        q_low = interpolate(self.net_integrated_heating_capacity, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.177
        p_low = interpolate(self.net_integrated_heating_power, self.H0_low_cond, self.H1_low_cond, t) # eq. 11.178
        cop_low = q_low/p_low
        q_int = q_H2_int + M_Hq*(t - (fr_u(35,"°F")))
        p_int = p_H2_int + M_HE*(t - (fr_u(35,"°F")))
        cop_int = q_int/p_int

        if bl <= q_low:
          if t <= self.heating_off_temperature or cop_low < 1.0:
            delta_low = 0.0 # eq. 11.159
          elif t > self.heating_on_temperature:
            delta_low = 1.0 # eq. 11.160
          else:
            delta_low = 0.5 # eq. 11.161

          hlf_low = bl/q_low # eq. 11.155
          plf_low = 1.0 - self.c_d_heating*(1.0 - hlf_low) # eq. 11.156
          e = p_low*hlf_low*delta_low*n/plf_low # eq. 11.153
          rh = bl*(1.0 - delta_low)*n # eq. 11.154
        elif bl < q_full:
          if bl <= q_int:
            cop_int_bin = cop_low + (cop_int - cop_low)/(q_int - q_low)*(bl - q_low) # eq. 11.187 (2023)
          else: # if bl > q_int:
            cop_int_bin = cop_int + (cop_full - cop_int)/(q_full - q_int)*(bl - q_int) # eq. 11.188 (2023)
          if t <= self.heating_off_temperature or cop_int_bin < 1.0:
            delta_int_bin = 0.0 # eq. 11.196
          elif t > self.heating_on_temperature:
            delta_int_bin = 1.0 # eq. 11.198
          else:
            delta_int_bin = 0.5 # eq. 11.197
          rh = bl*(1.0 - delta_int_bin)*n
          q = bl*n
          e = q/cop_int_bin*delta_int_bin
        else: # if bl >= q_full:
          # TODO: allow no H4 conditions
          # Note: builds on previously defined q_full / p_full
          if t > fr_u(5,"°F") or t <= fr_u(17,"°F"):
            q_full = interpolate(self.net_integrated_heating_capacity, self.H4_full_cond, self.H3_full_cond, t) # eq. 11.203
            p_full = interpolate(self.net_integrated_heating_power, self.H4_full_cond, self.H3_full_cond, t) # eq. 11.204
          elif t < fr_u(5,"°F"):
            t_ratio = (t - fr_u(5.0,"°F"))/(fr_u(47,"°F") - fr_u(17.0,"°F"))
            q_full = q_H4_full + (q_H1_full - q_H3_full)*t_ratio # eq. 11.205
            p_full = p_H4_full + (p_H1_full - p_H3_full)*t_ratio # eq. 11.206
          hlf_full = 1.0 # eq. 11.170
          e = p_full*hlf_full*delta_full*n # eq. 11.168
          rh = (bl - q_full*hlf_full*delta_full)*n # eq. 11.169

      q_sum += n*bl
      e_sum += e
      rh_sum += rh

    t_test = max(self.defrost.period, fr_u(90,"min"))
    t_max  = min(self.defrost.max_time, fr_u(720.0,'min'))

    if self.defrost.control == DefrostControl.DEMAND:
      f_def = 1 + 0.03 * (1 - (t_test-fr_u(90.0,'min'))/(t_max-fr_u(90.0,'min'))) # eq. 11.129
    else:
      f_def = 1 # eq. 11.130

    hspf = q_sum/(e_sum + rh_sum) * f_def # eq. 11.133
    return to_u(hspf,'Btu/Wh')

  def print_cooling_info(self):
    print(f"SEER: {self.seer()}")
    for speed in range(self.number_of_input_stages):
      conditions = CoolingConditions(compressor_speed=speed)
      conditions.set_rated_air_flow(self.flow_rated_per_cap_cooling_rated[speed], self.net_total_cooling_capacity_rated[speed])
      print(f"Net cooling power for stage {speed + 1} : {self.net_cooling_power(conditions)}")
      print(f"Net cooling capacity for stage {speed + 1} : {self.net_total_cooling_capacity(conditions)}")
      print(f"Net cooling EER for stage {speed + 1} : {self.eer(conditions)}")
      print(f"Gross cooling COP for stage {speed + 1} : {self.gross_cooling_cop(conditions)}")
      print(f"Net SHR for stage {speed + 1} : {self.net_shr(conditions)}")

  def print_heating_info(self, region=4):
    print(f"HSPF (region {region}): {self.hspf(region)}")
    for speed in range(self.number_of_input_stages):
      conditions = HeatingConditions(compressor_speed=speed)
      conditions.set_rated_air_flow(self.flow_rated_per_cap_heating_rated[speed], self.net_total_cooling_capacity_rated[speed])
      print(f"Net heating power for stage {speed + 1} : {self.net_integrated_heating_power(conditions)}")
      print(f"Gross heating COP for stage {speed + 1} : {self.gross_integrated_heating_cop(conditions)}")
      print(f"Net heating capacity for stage {speed + 1} : {self.net_integrated_heating_capacity(conditions)}")

  def write_A205(self):
    '''TODO: Write ASHRAE 205 file!!!'''
    return
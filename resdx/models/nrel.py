from scipy import optimize
import copy
from enum import Enum

from ..units import fr_u, to_u
from ..util import calc_biquad, calc_quad
from ..psychrometrics import psychrolib, PsychState
from ..defrost import DefrostControl, DefrostStrategy
from ..conditions import CoolingConditions, HeatingConditions
from ..fan import Fan

from .base_model import DXModel

class NRELDXModel(DXModel):

  '''Based on Cutler et al, but also includes internal EnergyPlus calculations'''
  '''Also, some assumptions from: https://github.com/NREL/OpenStudio-ERI/blob/master/hpxml-measures/HPXMLtoOpenStudio/resources/hvac.rb'''

  def gross_cooling_power(self, conditions):
    '''From Cutler et al.'''
    T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
    eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
    eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.mass_airflow_ratio)
    cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
    cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_airflow_ratio)
    return eir_FF*cap_FF*eir_FT*cap_FT*self.system.rated_gross_total_cooling_capacity[conditions.compressor_speed]/self.system.rated_gross_cooling_cop[conditions.compressor_speed]

  def gross_total_cooling_capacity(self, conditions):
    '''From Cutler et al.'''
    T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
    cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)  # Note: Equals 0.9915 at rating conditions (not 1.0)
    cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_airflow_ratio)
    return cap_FF*cap_FT*self.system.rated_gross_total_cooling_capacity[conditions.compressor_speed]

  def gross_steady_state_heating_power(self, conditions):
    '''From Cutler et al.'''
    T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
    eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
    eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.mass_airflow_ratio)
    cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
    cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_airflow_ratio)
    return eir_FF*cap_FF*eir_FT*cap_FT*self.system.rated_gross_heating_capacity[conditions.compressor_speed]/self.system.rated_gross_heating_cop[conditions.compressor_speed]

  def gross_steady_state_heating_capacity(self, conditions):
    '''From Cutler et al.'''
    T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
    cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
    cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_airflow_ratio)
    return cap_FF*cap_FT*self.system.rated_gross_heating_capacity[conditions.compressor_speed]

  def gross_integrated_heating_capacity(self, conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if self.system.defrost.in_defrost(conditions):
      t_defrost = self.system.defrost.time_fraction(conditions)
      if self.system.defrost.control ==DefrostControl.TIMED:
          heating_capacity_multiplier = 0.909 - 107.33*NRELDXModel.coil_diff_outdoor_air_humidity(conditions)
      else:
          heating_capacity_multiplier = 0.875*(1 - t_defrost)

      if self.system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
          Q_defrost_indoor_u = 0.01*(7.222 - to_u(conditions.outdoor.db,"°C"))*(self.system.rated_gross_heating_capacity[conditions.compressor_speed]/1.01667)
      else:
          Q_defrost_indoor_u = 0

      Q_with_frost_indoor_u = self.system.gross_steady_state_heating_capacity(conditions)*heating_capacity_multiplier
      return Q_with_frost_indoor_u*(1 - t_defrost) - Q_defrost_indoor_u*t_defrost
    else:
      return self.system.gross_steady_state_heating_capacity(conditions)

  def get_cooling_cop60(self, conditions):
    if "cooling_cop60" not in self.system.model_data:
      self.system.model_data["cooling_cop60"] = [None]*self.system.number_of_input_stages

    if self.system.model_data["cooling_cop60"][conditions.compressor_speed] is not None:
      return self.system.model_data["cooling_cop60"][conditions.compressor_speed]
    else:
      condition = self.system.make_condition(CoolingConditions,
                                        outdoor=PsychState(drybulb=fr_u(60.0,"°F"),wetbulb=fr_u(48.0,"°F")),  # 60 F at ~40% RH
                                        indoor=PsychState(drybulb=fr_u(70.0,"°F"),wetbulb=fr_u(60.0,"°F")),  # Use H1 indoor conditions (since we're still heating)
                                        compressor_speed=conditions.compressor_speed)
      self.system.model_data["cooling_cop60"][conditions.compressor_speed] = self.system.gross_cooling_cop(condition)
      return self.system.model_data["cooling_cop60"][conditions.compressor_speed]

  def gross_integrated_heating_power(self, conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if self.system.defrost.in_defrost(conditions):
      t_defrost = self.system.defrost.time_fraction(conditions)
      if self.system.defrost.control == DefrostControl.TIMED:
        input_power_multiplier = 0.9 - 36.45*NRELDXModel.coil_diff_outdoor_air_humidity(conditions)
      else:
        input_power_multiplier = 0.954*(1 - t_defrost)

      if self.system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
        #T_iwb = to_u(conditions.indoor.wb,"°C")
        #T_odb = conditions.outdoor.db_C
        # defEIRfT = calc_biquad([0.1528, 0, 0, 0, 0, 0], T_iwb, T_odb) # Assumption from BEopt 0.1528 = 1/gross_cop_cooling(60F)
        defEIRfT = 1/NRELDXModel.get_cooling_cop60(conditions)  # Assume defrost EIR is constant (maybe it could/should change with indoor conditions?)
        P_defrost = defEIRfT*(self.system.rated_gross_heating_capacity[conditions.compressor_speed]/1.01667)
      else:
        P_defrost = self.system.defrost.resistive_power

      P_with_frost = self.system.gross_steady_state_heating_power(conditions)*input_power_multiplier
      return P_with_frost*(1 - t_defrost) + P_defrost*t_defrost
    else:
      return self.system.gross_steady_state_heating_power(conditions)

  def epri_defrost_time_fraction(conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    return 1/(1+(0.01446/NRELDXModel.coil_diff_outdoor_air_humidity(conditions)))

  def coil_diff_outdoor_air_humidity(conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    T_coil_outdoor = 0.82 * to_u(conditions.outdoor.db,"°C") - 8.589  # In C
    saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.outdoor.p) # pressure in Pa already
    humidity_diff = conditions.outdoor.get_hr() - saturated_air_himidity_ratio
    return max(1.0e-6, humidity_diff)

  def gross_sensible_cooling_capacity(self, conditions):
    '''EnergyPlus algorithm'''
    Q_t = self.system.gross_total_cooling_capacity(conditions)
    h_i = conditions.indoor.get_h()
    m_dot = conditions.mass_airflow
    h_ADP = h_i - Q_t/(m_dot*(1 - self.system.bypass_factor(conditions)))
    root_fn = lambda T_ADP : psychrolib.GetSatAirEnthalpy(T_ADP, conditions.indoor.p) - h_ADP
    T_ADP = optimize.newton(root_fn, conditions.indoor.db_C)
    w_ADP = psychrolib.GetSatHumRatio(T_ADP, conditions.indoor.p)
    h_sensible = psychrolib.GetMoistAirEnthalpy(conditions.indoor.db_C,w_ADP)
    return Q_t*(h_sensible - h_ADP)/(h_i - h_ADP)

  class FunctionType(Enum):
    CAPACITY = 1
    POWER = 2

  class ConditionsType(Enum):
    COOLING = 1
    HEATING = 2

  '''From Domanski.'''
  charge_coeffs = {
    ConditionsType.COOLING: {
      FunctionType.CAPACITY:
        [
          [-9.46E-01, 4.93E-02, -1.18E-03, -1.15E+00],
          [-1.63E-01, 1.14E-02, -2.10E-04, -1.40E-01]
        ],
      FunctionType.POWER:
        [
          [-3.13E-01, 1.15E-02, 2.66E-03, -1.16E-01],
          [2.19E-01, -5.01E-03, 9.89E-04, 2.84E-01]
        ]
    },
    ConditionsType.HEATING: {
      FunctionType.CAPACITY:
        [
          [-3.39E-02, 0., 2.03E-02, -2.62E+00],
          [-2.95E-03, 0., 7.38E-04, -6.41E-03]
        ],
      FunctionType.POWER:
        [
          [6.16E-02, 0., 4.46E-03, -2.60E-01],
          [-5.94E-02, 0., 1.59E-02, 1.89E+00]
        ]
    }
  }

  @staticmethod
  def domonski_charge_factor(T_idb, T_odb, f_chg, coeffs):
    if f_chg < 0:
      return 1 + (coeffs[0][0] + coeffs[0][1]*T_idb + coeffs[0][2]*T_odb + coeffs[0][3]*f_chg)*f_chg
    else:
      return 1 + (coeffs[1][0] + coeffs[1][1]*T_idb + coeffs[1][2]*T_odb + coeffs[1][3]*f_chg)*f_chg

  def resnet_grading_model(self, conditions, function_type):
    f_chg = self.system.refrigerant_charge_deviation
    if f_chg == 0.:
      return 1.0
    conditions_class = type(conditions)
    if conditions_class == HeatingConditions:
      conditions_type = NRELDXModel.ConditionsType.HEATING
    else:
      conditions_type = NRELDXModel.ConditionsType.COOLING

    if conditions_type == NRELDXModel.ConditionsType.COOLING:
      if function_type == NRELDXModel.FunctionType.CAPACITY:
        function = self.gross_total_cooling_capacity
      else:
        function = self.gross_cooling_power
    else:
      if function_type == NRELDXModel.FunctionType.CAPACITY:
        function = self.gross_steady_state_heating_capacity
      else:
        function = self.gross_steady_state_heating_power

    coeffs = NRELDXModel.charge_coeffs[conditions_type][function_type]

    rated_cond = self.system.make_condition(conditions_class,compressor_speed=conditions.compressor_speed)
    rated_value = function(rated_cond)
    f = NRELDXModel.domonski_charge_factor(conditions.indoor.db_C, conditions.outdoor.db_C, f_chg, coeffs)

    # f_af_chg = X_AF,CHG. This makes the calculation of the AF,CHG fault independent of the airflow correction method used in the model.
    cf_af_chg = 1./NRELDXModel.domonski_charge_factor(rated_cond.indoor.db_C, rated_cond.outdoor.db_C, f_chg, NRELDXModel.charge_coeffs[conditions_type][NRELDXModel.FunctionType.CAPACITY])
    cf_af_cond = copy.deepcopy(rated_cond)
    cf_af_cond.set_mass_airflow_ratio(cf_af_chg)
    f_af_chg = function(cf_af_cond)/rated_value

    # f_af = X_AF. Similar story
    f_af_comb = conditions.mass_airflow_ratio*cf_af_chg
    f_af_comb_cond = copy.deepcopy(rated_cond)
    f_af_comb_cond.set_mass_airflow_ratio(f_af_comb)
    f_af = function(f_af_comb_cond)/rated_value

    # grading model accounts for airflow sensitivity so we need to undo the normal affect
    f_corr_cond = copy.deepcopy(rated_cond)
    f_corr_cond.set_mass_airflow_ratio(conditions.mass_airflow_ratio)
    f_corr = function(f_corr_cond)/rated_value

    return f/f_af_chg*f_af/f_corr


  def gross_total_cooling_capacity_charge_factor(self, conditions):
    return NRELDXModel.resnet_grading_model(self, conditions, NRELDXModel.FunctionType.CAPACITY)

  def gross_cooling_power_charge_factor(self, conditions):
    return NRELDXModel.resnet_grading_model(self, conditions, NRELDXModel.FunctionType.POWER)

  def gross_steady_state_heating_capacity_charge_factor(self, conditions):
    return NRELDXModel.resnet_grading_model(self, conditions, NRELDXModel.FunctionType.CAPACITY)

  def gross_steady_state_heating_power_charge_factor(self, conditions):
    return NRELDXModel.resnet_grading_model(self, conditions, NRELDXModel.FunctionType.POWER)

  # Default assumptions
  def set_rated_fan_characteristics(self, fan):
    if fan is not None:
      pass
    else:
      if self.system.input_seer is not None:
        if self.system.input_seer <= 14.:
          fan_efficacy = fr_u(0.25,'W/cfm')
        elif self.system.input_seer >= 16.:
          fan_efficacy = fr_u(0.18,'W/cfm')
        else:
          fan_efficacy = fr_u(0.25,'W/cfm') + (fr_u(0.18,'W/cfm') - fr_u(0.25,'W/cfm')) * (self.system.input_seer - 14.0) / 2.0 # W/cfm
      else:
        fan_efficacy = fr_u(0.25,'W/cfm')
      self.system.rated_cooling_fan_efficacy = [fan_efficacy]*self.system.number_of_input_stages
      self.system.rated_heating_fan_efficacy = [fan_efficacy]*self.system.number_of_input_stages
      if self.system.number_of_input_stages == 1:
        self.system.rated_cooling_airflow_per_rated_net_capacity = [fr_u(394.2,"cfm/ton_ref")]
        self.system.rated_heating_airflow_per_rated_net_capacity = [fr_u(384.1,"cfm/ton_ref")]
      elif self.system.number_of_input_stages == 2:
        cooling_default = fr_u(344.1,"cfm/ton_ref")
        self.system.rated_cooling_airflow_per_rated_net_capacity = [cooling_default, cooling_default*0.86]
        heating_default = fr_u(352.2,"cfm/ton_ref")
        self.system.rated_heating_airflow_per_rated_net_capacity = [heating_default, heating_default*0.8]

  def set_rated_net_total_cooling_capacity(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_total_cooling_capacity = input
    else:
      if self.system.number_of_input_stages == 1:
        self.system.rated_net_total_cooling_capacity = [input]
      elif self.system.number_of_input_stages == 2:
        cap_ratio = 0.72
        fan_power_0 = input*self.system.rated_cooling_fan_efficacy[0]*self.system.rated_cooling_airflow_per_rated_net_capacity[0]
        gross_cap_0 = input + fan_power_0
        gross_cap_1 = gross_cap_0*cap_ratio
        net_cap_1 = gross_cap_1/(1. + self.system.rated_cooling_fan_efficacy[1]*self.system.rated_cooling_airflow_per_rated_net_capacity[1])
        self.system.rated_net_total_cooling_capacity = [input, net_cap_1]

  def set_rated_net_heating_capacity(self, input):
    input = self.set_default(input, self.system.rated_net_total_cooling_capacity[0])
    if type(input) is list:
      self.system.rated_net_heating_capacity = input
    else:
      if self.system.number_of_input_stages == 1:
        self.system.rated_net_heating_capacity = [input]
      elif self.system.number_of_input_stages == 2:
        cap_ratio = 0.72
        fan_power_0 = input*self.system.rated_heating_fan_efficacy[0]*self.system.rated_heating_airflow_per_rated_net_capacity[0]
        gross_cap_0 = input - fan_power_0
        gross_cap_1 = gross_cap_0*cap_ratio
        net_cap_1 = gross_cap_1/(1. - self.system.rated_heating_fan_efficacy[1]*self.system.rated_heating_airflow_per_rated_net_capacity[1])
        self.system.rated_net_heating_capacity = [input, net_cap_1]

  def set_fan(self, input):
    if input is not None:
      # TODO: Handle default mappings?
      self.system.fan = input
    else:
      airflows = []
      efficacies = []
      fan_speed = 0
      if self.system.cooling_fan_speed is None:
        set_cooling_fan_speed = True
        self.system.cooling_fan_speed = []

      if self.system.heating_fan_speed is None:
        set_heating_fan_speed = True
        self.system.heating_fan_speed = []

      for i, cap in enumerate(self.system.rated_net_total_cooling_capacity):
        airflows.append(cap*self.system.rated_cooling_airflow_per_rated_net_capacity[i])
        efficacies.append(self.system.rated_cooling_fan_efficacy[i])
        if set_cooling_fan_speed:
          self.system.cooling_fan_speed.append(fan_speed)
          fan_speed += 1

      for i, cap in enumerate(self.system.rated_net_total_cooling_capacity):
        airflows.append(cap*self.system.rated_heating_airflow_per_rated_net_capacity[i])
        efficacies.append(self.system.rated_heating_fan_efficacy[i])
        if set_heating_fan_speed:
          self.system.heating_fan_speed.append(fan_speed)
          fan_speed += 1

      fan = NRELFan(airflows, fr_u(0.20, "in_H2O"), design_efficacy=efficacies)
      self.system.fan = fan

  def set_net_capacities_and_fan(self, rated_net_total_cooling_capacity, rated_net_heating_capacity, fan):
    self.set_rated_fan_characteristics(fan)
    self.set_rated_net_total_cooling_capacity(rated_net_total_cooling_capacity)
    self.set_rated_net_heating_capacity(rated_net_heating_capacity)
    self.set_fan(fan)

  @staticmethod
  def cooling_cop_low(cooling_cop_high):
    eir_high = 1./cooling_cop_high
    eir_low = 0.8887 * (eir_high) + 0.0083
    return 1./eir_low

  @staticmethod
  def heating_cop_low(heating_cop_high):
    eir_high = 1./heating_cop_high
    eir_low = 0.6241 * (eir_high) + 0.0681
    return 1./eir_low

  def set_rated_net_cooling_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_cooling_cop = input
      self.system.rated_net_cooling_power = [self.system.rated_net_total_cooling_capacity[i]/self.system.rated_net_cooling_cop[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_gross_cooling_power = [self.system.rated_net_cooling_power[i] - self.system.rated_cooling_fan_power[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_gross_cooling_cop = [self.system.rated_gross_total_cooling_capacity[i]/self.system.rated_gross_cooling_power[i] for i in range(self.system.number_of_input_stages)]
    else:
      self.system.rated_net_cooling_cop[0] = input
      self.system.rated_net_cooling_power[0] = self.system.rated_net_total_cooling_capacity[0]/self.system.rated_net_cooling_cop[0]
      self.system.rated_gross_cooling_power[0] = self.system.rated_net_cooling_power[0] - self.system.rated_cooling_fan_power[0]
      self.system.rated_gross_cooling_cop[0] = self.system.rated_gross_total_cooling_capacity[0]/self.system.rated_gross_cooling_power[0]

      if self.system.number_of_input_stages == 2:
        self.system.rated_gross_cooling_cop[1] = NRELDXModel.cooling_cop_low(self.system.rated_gross_cooling_cop[0])
        self.system.rated_gross_cooling_power[1] = self.system.rated_gross_total_cooling_capacity[1]/self.system.rated_gross_cooling_cop[1]
        self.system.rated_net_cooling_power[1] = self.system.rated_gross_cooling_power[1] + self.system.rated_cooling_fan_power[1]
        self.system.rated_net_cooling_cop[1] = self.system.rated_net_total_cooling_capacity[1]/self.system.rated_net_cooling_power[1]

  def set_rated_gross_cooling_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_gross_cooling_cop = input
    else:
      self.system.rated_gross_cooling_cop[0] = input
      if self.system.number_of_input_stages == 2:
        self.system.rated_gross_cooling_cop[1] = NRELDXModel.cooling_cop_low(self.system.rated_gross_cooling_cop[0])

    self.system.rated_gross_cooling_power = [self.system.rated_gross_total_cooling_capacity[i]/self.system.rated_gross_cooling_cop[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_net_cooling_power = [self.system.rated_gross_cooling_power[i] + self.system.rated_cooling_fan_power[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_net_cooling_cop = [self.system.rated_net_total_cooling_capacity[i]/self.system.rated_net_cooling_power[i] for i in range(self.system.number_of_input_stages)]

  def set_rated_net_heating_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_heating_cop = input
      self.system.rated_net_heating_power = [self.system.rated_net_heating_capacity[i]/self.system.rated_net_heating_cop[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_gross_heating_power = [self.system.rated_net_heating_power[i] - self.system.rated_heating_fan_power[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_gross_heating_cop = [self.system.rated_gross_heating_capacity[i]/self.system.rated_gross_heating_power[i] for i in range(self.system.number_of_input_stages)]
    else:
      self.system.rated_net_heating_cop[0] = input
      self.system.rated_net_heating_power[0] = self.system.rated_net_heating_capacity[0]/self.system.rated_net_heating_cop[0]
      self.system.rated_gross_heating_power[0] = self.system.rated_net_heating_power[0] - self.system.rated_heating_fan_power[0]
      self.system.rated_gross_heating_cop[0] = self.system.rated_gross_heating_capacity[0]/self.system.rated_gross_heating_power[0]

      if self.system.number_of_input_stages == 2:
        self.system.rated_gross_heating_cop[1] = NRELDXModel.heating_cop_low(self.system.rated_gross_heating_cop[0])
        self.system.rated_gross_heating_power[1] = self.system.rated_gross_heating_capacity[1]/self.system.rated_gross_heating_cop[1]
        self.system.rated_net_heating_power[1] = self.system.rated_gross_heating_power[1] + self.system.rated_heating_fan_power[1]
        self.system.rated_net_heating_cop[1] = self.system.rated_net_heating_capacity[1]/self.system.rated_net_heating_power[1]

  def set_rated_gross_heating_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_gross_heating_cop = input
    else:
      self.system.rated_gross_heating_cop[0] = input
      if self.system.number_of_input_stages == 2:
        self.system.rated_gross_heating_cop[1] = NRELDXModel.heating_cop_low(self.system.rated_gross_heating_cop[0])

    self.system.rated_gross_heating_power = [self.system.rated_gross_heating_capacity[i]/self.system.rated_gross_heating_cop[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_net_heating_power = [self.system.rated_gross_heating_power[i] + self.system.rated_heating_fan_power[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_net_heating_cop = [self.system.rated_net_heating_capacity[i]/self.system.rated_net_heating_power[i] for i in range(self.system.number_of_input_stages)]

# Fan

class NRELFan(Fan):
  def __init__(
    self,
    design_airflow,
    design_external_static_pressure=fr_u(0.5, "in_H2O"),
    design_efficacy=fr_u(0.365,'W/cfm')):
      self.power_ratio = []
      self.design_power = []
      super().__init__(design_airflow, design_external_static_pressure, design_efficacy)

  def add_speed(self, airflow):
    super().add_speed(airflow)
    self.power_ratio.append(self.design_airflow_ratio[-1]**3.)
    self.design_power.append(self.design_airflow[0]*self.design_efficacy*self.power_ratio[-1])

  def remove_speed(self, speed_setting):
    super().remove_speed(speed_setting)
    self.power_ratio.pop(speed_setting)
    self.design_power.pop(speed_setting)

  def airflow(self, conditions):
    return self.design_airflow[conditions.speed_setting]

  def power(self, conditions):
    return self.design_power[conditions.speed_setting]

  def efficacy(self, conditions):
    return self.power(conditions)/self.airflow(conditions)
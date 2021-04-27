from scipy import optimize

from ..units import fr_u, to_u
from ..util import calc_biquad, calc_quad
from ..psychrometrics import psychrolib, PsychState
from ..defrost import DefrostControl, DefrostStrategy
from ..conditions import CoolingConditions

from .base_model import DXModel

class ConstantDXModel(DXModel):

  '''This model considers the HP performance independant from weather conditions'''

  @staticmethod
  def gross_cooling_power(conditions, system):
    '''From Cutler et al.'''
    return system.gross_total_cooling_capacity_rated[conditions.compressor_speed]/system.gross_cooling_cop_rated[conditions.compressor_speed]

  @staticmethod
  def gross_total_cooling_capacity(conditions, system):
    '''From Cutler et al.'''
    return system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    '''From Cutler et al.'''
    return system.gross_heating_capacity_rated[conditions.compressor_speed]/system.gross_heating_cop_rated[conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_capacity(conditions, system):
    '''From Cutler et al.'''
    return system.gross_heating_capacity_rated[conditions.compressor_speed]

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if system.defrost.in_defrost(conditions):
      t_defrost = system.defrost.time_fraction(conditions)
      if system.defrost.control ==DefrostControl.TIMED:
          heating_capacity_multiplier = 0.909 - 107.33*ConstantDXModel.coil_diff_outdoor_air_humidity(conditions)
      else:
          heating_capacity_multiplier = 0.875*(1 - t_defrost)

      if system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
          Q_defrost_indoor_u = 0.01*(7.222 - to_u(conditions.outdoor.db,"°C"))*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
      else:
          Q_defrost_indoor_u = 0

      Q_with_frost_indoor_u = system.gross_steady_state_heating_capacity(conditions)*heating_capacity_multiplier
      return Q_with_frost_indoor_u*(1 - t_defrost) - Q_defrost_indoor_u*t_defrost
    else:
      return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_heating_capacity_with_frost(conditions, system): # This should be equivalent to the output variable DX heating coil heating rate [W] in EnergyPlus
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if system.defrost.in_defrost(conditions):
      t_defrost = system.defrost.time_fraction(conditions)
      if system.defrost.control ==DefrostControl.TIMED:
          heating_capacity_multiplier = 0.909 - 107.33*ConstantDXModel.coil_diff_outdoor_air_humidity(conditions)
      else:
          heating_capacity_multiplier = 0.875*(1 - t_defrost)

      Q_with_frost_indoor_u = system.gross_steady_state_heating_capacity(conditions)*heating_capacity_multiplier
      return Q_with_frost_indoor_u
    else:
      return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_heating_capacity_defrost_indoor(conditions, system):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if system.defrost.in_defrost(conditions):
      if system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
          Q_defrost_indoor_u = 0.01*(7.222 - to_u(conditions.outdoor.db,"°C"))*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
      else:
          Q_defrost_indoor_u = 0

      return Q_defrost_indoor_u
    else:
      return 0.0


  @staticmethod
  def get_cooling_cop60(conditions, system):
    if "cooling_cop60" not in system.model_data:
      system.model_data["cooling_cop60"] = [None]*system.number_of_speeds

    if system.model_data["cooling_cop60"][conditions.compressor_speed] is not None:
      return system.model_data["cooling_cop60"][conditions.compressor_speed]
    else:
      condition = system.make_condition(CoolingConditions,
                                        outdoor=PsychState(drybulb=fr_u(60.0,"°F"),wetbulb=fr_u(48.0,"°F")),  # 60 F at ~40% RH
                                        indoor=PsychState(drybulb=fr_u(70.0,"°F"),wetbulb=fr_u(60.0,"°F")),  # Use H1 indoor conditions (since we're still heating)
                                        compressor_speed=conditions.compressor_speed)
      system.model_data["cooling_cop60"][conditions.compressor_speed] = system.gross_cooling_cop(condition)
      return system.model_data["cooling_cop60"][conditions.compressor_speed]

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    if system.defrost.in_defrost(conditions):
      t_defrost = system.defrost.time_fraction(conditions)
      if system.defrost.control == DefrostControl.TIMED:
        input_power_multiplier = 0.9 - 36.45*ConstantDXModel.coil_diff_outdoor_air_humidity(conditions)
      else:
        input_power_multiplier = 0.954*(1 - t_defrost)

      if system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
        #T_iwb = to_u(conditions.indoor.wb,"°C")
        #T_odb = conditions.outdoor.db_C
        # defEIRfT = calc_biquad([0.1528, 0, 0, 0, 0, 0], T_iwb, T_odb) # Assumption from BEopt 0.1528 = 1/gross_cop_cooling(60F)
        defEIRfT = 0.16 #1/ConstantDXModel.get_cooling_cop60(conditions, system)  # Assume defrost EIR is constant (maybe it could/should change with indoor conditions?)
        P_defrost = defEIRfT*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
      else:
        P_defrost = system.defrost.resistive_power

      P_with_frost = system.gross_steady_state_heating_power(conditions)*input_power_multiplier
      return P_with_frost*(1 - t_defrost) + P_defrost*t_defrost
    else:
      return system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def epri_defrost_time_fraction(conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    return 1/(1+(0.01446/ConstantDXModel.coil_diff_outdoor_air_humidity(conditions)))

  @staticmethod
  def coil_diff_outdoor_air_humidity(conditions):
    '''EPRI algorithm as described in EnergyPlus documentation'''
    T_coil_outdoor = 0.82 * to_u(conditions.outdoor.db,"°C") - 8.589  # In C
    saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.outdoor.p) # pressure in Pa already
    humidity_diff = conditions.outdoor.get_hr() - saturated_air_himidity_ratio
    return max(1.0e-6, humidity_diff)

  @staticmethod
  def gross_sensible_cooling_capacity(conditions, system):
    '''EnergyPlus algorithm'''
    Q_t = system.gross_total_cooling_capacity(conditions)
    h_i = conditions.indoor.get_h()
    m_dot = conditions.air_mass_flow
    h_ADP = h_i - Q_t/(m_dot*(1 - system.bypass_factor(conditions)))
    root_fn = lambda T_ADP : psychrolib.GetSatAirEnthalpy(T_ADP, conditions.indoor.p) - h_ADP
    T_ADP = optimize.newton(root_fn, conditions.indoor.db_C)
    w_ADP = psychrolib.GetSatHumRatio(T_ADP, conditions.indoor.p)
    h_sensible = psychrolib.GetMoistAirEnthalpy(conditions.indoor.db_C,w_ADP)
    return Q_t*(h_sensible - h_ADP)/(h_i - h_ADP)

  @staticmethod
  def gross_shr(conditions):
    # not used because in this example file we are interested in heating only.
    return 0.7 # Higher values will lead to a negative BF_rated => error.



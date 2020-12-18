from scipy import optimize

from .util import calc_biquad, calc_quad
from .psychrometrics import psychrolib
from .units import fr_u, to_u

from .defrost import Defrost, DefrostControl, DefrostStrategy

## Model functions

# NREL Model
'''Based on Cutler et al, but also includes internal EnergyPlus calculations'''

def cutler_cooling_power(conditions, power_scalar):
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*power_scalar

def cutler_total_cooling_capacity(conditions, capacity_scalar):
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*capacity_scalar

def cutler_steady_state_heating_power(conditions, power_scalar):
  T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*power_scalar

def cutler_steady_state_heating_capacity(conditions, capacity_scalar):
  T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*capacity_scalar

def epri_integrated_heating_capacity(conditions, capacity_scalar, defrost):
  # EPRI algorithm as described in EnergyPlus documentation
  if defrost.in_defrost(conditions):
    t_defrost = defrost.time_fraction(conditions)
    if defrost.control ==DefrostControl.TIMED:
        heating_capacity_multiplier = 0.909 - 107.33*coil_diff_outdoor_air_humidity(conditions)
    else:
        heating_capacity_multiplier = 0.875*(1 - t_defrost)

    if defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
        Q_defrost_indoor_u = 0.01*(7.222 - to_u(conditions.outdoor.db,"°C"))*(capacity_scalar/1.01667)
    else:
        Q_defrost_indoor_u = 0

    Q_with_frost_indoor_u = cutler_steady_state_heating_capacity(conditions,capacity_scalar)*heating_capacity_multiplier
    return Q_with_frost_indoor_u*(1 - t_defrost) - Q_defrost_indoor_u*t_defrost
  else:
    return cutler_steady_state_heating_capacity(conditions,capacity_scalar)

def epri_integrated_heating_power(conditions, power_scalar, capacity_scalar, defrost):
  # EPRI algorithm as described in EnergyPlus documentation
  if defrost.in_defrost(conditions):
    t_defrost = defrost.time_fraction(conditions)
    if defrost.control == DefrostControl.TIMED:
      input_power_multiplier = 0.9 - 36.45*coil_diff_outdoor_air_humidity(conditions)
    else:
      input_power_multiplier = 0.954*(1 - t_defrost)

    if defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
      T_iwb = to_u(conditions.indoor.wb,"°C")
      T_odb = conditions.outdoor.db_C
      defEIRfT = calc_biquad([0.1528, 0, 0, 0, 0, 0], T_iwb, T_odb) # TODO: Check assumption from BEopt
      P_defrost = defEIRfT*(capacity_scalar/1.01667)
    else:
      P_defrost = defrost.resistive_power

    P_with_frost = cutler_steady_state_heating_power(conditions,power_scalar)*input_power_multiplier
    return P_with_frost*(1 - t_defrost) + P_defrost*t_defrost
  else:
    return cutler_steady_state_heating_power(conditions,power_scalar)

def epri_defrost_time_fraction(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  return 1/(1+(0.01446/coil_diff_outdoor_air_humidity(conditions)))

def coil_diff_outdoor_air_humidity(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  T_coil_outdoor = 0.82 * to_u(conditions.outdoor.db,"°C") - 8.589  # In C
  saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.outdoor.p) # pressure in Pa already
  humidity_diff = conditions.outdoor.get_hr() - saturated_air_himidity_ratio
  return max(1.0e-6, humidity_diff)

def energyplus_sensible_cooling_capacity(conditions,total_capacity,bypass_factor):
  Q_t = total_capacity
  h_i = conditions.indoor.get_h()
  m_dot = conditions.air_mass_flow
  h_ADP = h_i - Q_t/(m_dot*(1 - bypass_factor))
  root_fn = lambda T_ADP : psychrolib.GetSatAirEnthalpy(T_ADP, conditions.indoor.p) - h_ADP
  T_ADP = optimize.newton(root_fn, conditions.indoor.db_C)
  w_ADP = psychrolib.GetSatHumRatio(T_ADP, conditions.indoor.p)
  h_sensible = psychrolib.GetMoistAirEnthalpy(conditions.indoor.db_C,w_ADP)
  return Q_t*(h_sensible - h_ADP)/(h_i - h_ADP)

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
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  T_idb = to_u(conditions.indoor.db,"°F") # Title 24 curves use °F
  CFM_per_ton = to_u(conditions.std_air_vol_flow_per_capacity,"cu_ft/min/ton_of_refrigeration")
  coeffs = [0.0242020,-0.0592153,0.0012651,0.0016375,0,0,0,-0.0000165,0,0.0002021,0,1.5085285]
  SHR = CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
  return min(1.0, SHR)


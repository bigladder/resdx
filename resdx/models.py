from scipy import optimize

from .util import calc_biquad, calc_quad
from .psychrometrics import psychrolib
from .units import u, convert

from .defrost import Defrost, DefrostControl, DefrostStrategy

## Model functions

# NREL Model
'''Based on Cutler et al, but also includes internal EnergyPlus calculations'''

def cutler_gross_cooling_power(conditions, system):
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]/system.gross_cooling_cop_rated[conditions.compressor_speed]

def cutler_gross_total_cooling_capacity(conditions, system):
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

def cutler_gross_steady_state_heating_power(conditions, system):
  T_idb = convert(conditions.indoor.db,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*system.gross_heating_capacity_rated[conditions.compressor_speed]/system.gross_heating_cop_rated[conditions.compressor_speed]

def cutler_gross_steady_state_heating_capacity(conditions, system):
  T_idb = convert(conditions.indoor.db,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*system.gross_heating_capacity_rated[conditions.compressor_speed]

def epri_gross_integrated_heating_capacity(conditions, system):
  # EPRI algorithm as described in EnergyPlus documentation
  if system.defrost.in_defrost(conditions):
    t_defrost = system.defrost.time_fraction(conditions)
    if system.defrost.control ==DefrostControl.TIMED:
        heating_capacity_multiplier = 0.909 - 107.33*coil_diff_outdoor_air_humidity(conditions)
    else:
        heating_capacity_multiplier = 0.875*(1 - t_defrost)

    if system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
        Q_defrost_indoor_u = 0.01*(7.222 - convert(conditions.outdoor.db,"°K","°C"))*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
    else:
        Q_defrost_indoor_u = 0

    Q_with_frost_indoor_u = system.gross_steady_state_heating_capacity(conditions)*heating_capacity_multiplier
    return Q_with_frost_indoor_u*(1 - t_defrost) - Q_defrost_indoor_u*t_defrost
  else:
    return system.gross_steady_state_heating_capacity(conditions)

def epri_gross_integrated_heating_power(conditions, system):
  # EPRI algorithm as described in EnergyPlus documentation
  if system.defrost.in_defrost(conditions):
    t_defrost = system.defrost.time_fraction(conditions)
    if system.defrost.control == DefrostControl.TIMED:
      input_power_multiplier = 0.9 - 36.45*coil_diff_outdoor_air_humidity(conditions)
    else:
      input_power_multiplier = 0.954*(1 - t_defrost)

    if system.defrost.strategy == DefrostStrategy.REVERSE_CYCLE:
      T_iwb = convert(conditions.indoor.wb,"K","°C")
      T_odb = conditions.outdoor.db_C
      defEIRfT = calc_biquad([0.1528, 0, 0, 0, 0, 0], T_iwb, T_odb) # TODO: Check assumption from BEopt
      P_defrost = defEIRfT*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
    else:
      P_defrost = system.defrost.resistive_power

    P_with_frost = system.gross_steady_state_heating_power(conditions)*input_power_multiplier
    return P_with_frost*(1 - t_defrost) + P_defrost*t_defrost
  else:
    return system.gross_steady_state_heating_power(conditions)

def epri_defrost_time_fraction(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  return 1/(1+(0.01446/coil_diff_outdoor_air_humidity(conditions)))

def coil_diff_outdoor_air_humidity(conditions):
  # EPRI algorithm as described in EnergyPlus documentation
  T_coil_outdoor = 0.82 * convert(conditions.outdoor.db,"°K","°C") - 8.589  # In C
  saturated_air_himidity_ratio = psychrolib.GetSatHumRatio(T_coil_outdoor,conditions.outdoor.p) # pressure in Pa already
  humidity_diff = conditions.outdoor.get_hr() - saturated_air_himidity_ratio
  return max(1.0e-6, humidity_diff)

def energyplus_gross_sensible_cooling_capacity(conditions, system):
  Q_t = system.gross_total_cooling_capacity(conditions)
  h_i = conditions.indoor.get_h()
  m_dot = conditions.air_mass_flow
  h_ADP = h_i - Q_t/(m_dot*(1 - system.bypass_factor(conditions)))
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
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Title 24 curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Title 24 curves use °F
  T_idb = convert(conditions.indoor.db,"K","°F") # Title 24 curves use °F
  CFM_per_ton = convert(conditions.std_air_vol_flow_per_capacity,"m**3/W/s","cu_ft/min/ton_of_refrigeration")
  coeffs = [0.0242020,-0.0592153,0.0012651,0.0016375,0,0,0,-0.0000165,0,0.0002021,0,1.5085285]
  SHR = CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
  return min(1.0, SHR)

def title24_eer_rated(seer):
  if seer < 13.0:
    return 10.0  + 0.84 * (seer - 11.5)
  elif seer < 16.0:
    return 11.3 + 0.57 * (seer - 13.0)
  else:
    return 13.0

def title24_gross_total_cooling_capacity(conditions, system):
  shr = title24_shr(conditions)
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Title 24 curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Title 24 curves use °F
  T_idb = convert(conditions.indoor.db,"K","°F") # Title 24 curves use °F
  CFM_per_ton = convert(conditions.std_air_vol_flow_per_capacity,"m**3/W/s","cu_ft/min/ton_of_refrigeration")
  if shr < 1:
    coeffs = [0,0.009645900,0.002536900,0.000171500,0,0,-0.000095900,0.000008180,-0.000007550,0.000105700,-53.542300000,0.381567150]
  else: # shr == 1
    coeffs = [0.009483100,0,-0.000600600,-0.000148900,-0.000032600,0.000011900,0,0,-0.000005050,0,-52.561740000,0.430751600]
  return CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

def title24_gross_sensible_cooling_capacity(conditions, system):
  return title24_shr(conditions)*system.gross_total_cooling_capacity(conditions)

def title24_gross_cooling_power(conditions, system):
  shr = title24_shr(conditions)
  T_iwb = convert(conditions.indoor.get_wb(),"K","°F") # Title 24 curves use °F
  T_odb = convert(conditions.outdoor.db,"K","°F") # Title 24 curves use °F
  T_idb = convert(conditions.indoor.db,"K","°F") # Title 24 curves use °F
  CFM_per_ton = convert(conditions.std_air_vol_flow_per_capacity,"m**3/W/s","cu_ft/min/ton_of_refrigeration")
  cap95 = system.net_total_cooling_capacity_rated[conditions.compressor_speed]
  q_fan = system.cooling_fan_power_rated[conditions.compressor_speed]
  if T_odb < 95.0:
    seer = u(system.kwargs["input_seer"],'Btu/Wh')
    if shr < 1:
      seer_coeffs = [0,-0.0202256,0.0236703,-0.0006638,0,0,-0.0001841,0.0000214,-0.00000812,0.0002971,-27.95672,0.209951063]
    else: # shr == 1
      seer_coeffs = [0.0046103,0,0.0125598,-0.000512,-0.0000357,0.0000105,0,0,0,0,0,-0.316172311]
    f_cond_seer = CA_regression(seer_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
    seer_nf = f_cond_seer*(1.09*cap95+q_fan)/(1.09*cap95/seer - q_fan) # unitless
  else:
    seer_nf = 0.0
  if T_odb > 82.0:
    if "input_eer" in system.kwargs:
      eer = u(system.kwargs["input_eer"],'Btu/Wh')
    else:
      eer = u(title24_eer_rated(system.kwargs["input_seer"]),'Btu/Wh')
    if shr < 1:
      eer_coeffs = [0,-0.020225600,0.023670300,-0.000663800,0,0,-0.000184100,0.000021400,-0.000008120,0.000297100,-27.956720000,0.015003100]
    else: # shr == 1
      eer_coeffs = [0.004610300,0,0.012559800,-0.000512000,-0.000035700,0.000010500,0,0,0,0,0,-0.475306500]
    cap_nf = system.gross_total_cooling_capacity_rated[conditions.compressor_speed]
    f_cond_eer = CA_regression(eer_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
    eer_nf = cap_nf/(f_cond_eer*(cap95/eer - q_fan/3.413))
  else:
    eer_nf = 0.0
  if T_odb <= 82.0:
    eer_t = seer_nf
  elif T_odb < 95.0:
    eer_t = seer_nf + (T_odb - 82.0)*(eer_nf - seer_nf)/13.0
  else:
    eer_t = eer_nf
  if "input_cooling_efficiency_multiplier" in system.kwargs:
    f_eff = system.kwargs["input_cooling_efficiency_multiplier"]
  else:
    f_eff = 1.0
  return system.gross_total_cooling_capacity(conditions)/(eer_t*f_eff)


# Unified RESNET Model
resnet_gross_cooling_power = cutler_gross_cooling_power
resnet_gross_total_cooling_capacity = cutler_gross_total_cooling_capacity
resnet_gross_sensible_cooling_capacity = energyplus_gross_sensible_cooling_capacity
resnet_shr_rated = title24_shr
resnet_gross_steady_state_heating_capacity = cutler_gross_steady_state_heating_capacity
resnet_gross_integrated_heating_capacity = epri_gross_integrated_heating_capacity
resnet_gross_steady_state_heating_power = cutler_gross_steady_state_heating_power
resnet_gross_integrated_heating_power = epri_gross_integrated_heating_power


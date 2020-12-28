from enum import Enum
from scipy import optimize

from .util import calc_biquad, calc_quad
from .psychrometrics import psychrolib
from .units import fr_u, to_u

from .defrost import Defrost, DefrostControl, DefrostStrategy

## Model functions

# NREL Model
'''Based on Cutler et al, but also includes internal EnergyPlus calculations'''

def cutler_gross_cooling_power(conditions, system):
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]/system.gross_cooling_cop_rated[conditions.compressor_speed]

def cutler_gross_total_cooling_capacity(conditions, system):
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.air_mass_flow_fraction)
  return cap_FF*cap_FT*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

def cutler_gross_steady_state_heating_power(conditions, system):
  T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.air_mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.air_mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*system.gross_heating_capacity_rated[conditions.compressor_speed]/system.gross_heating_cop_rated[conditions.compressor_speed]

def cutler_gross_steady_state_heating_capacity(conditions, system):
  T_idb = to_u(conditions.indoor.db,"°F") # Cutler curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Cutler curves use °F
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
        Q_defrost_indoor_u = 0.01*(7.222 - to_u(conditions.outdoor.db,"°C"))*(system.gross_heating_capacity_rated[conditions.compressor_speed]/1.01667)
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
      T_iwb = to_u(conditions.indoor.wb,"°C")
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
  T_coil_outdoor = 0.82 * to_u(conditions.outdoor.db,"°C") - 8.589  # In C
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

# FSEC Model (TODO)

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

def title24_eer_rated(seer):
  if seer < 13.0:
    return 10.0  + 0.84 * (seer - 11.5)
  elif seer < 16.0:
    return 11.3 + 0.57 * (seer - 13.0)
  else:
    return 13.0

class MotorType(Enum):
  PSC = 1,
  BPM = 2

def title24_fan_efficacy_rated(flow_per_capacity, motor_type=MotorType.PSC):
  if motor_type == MotorType.PSC:
    power_per_capacity = fr_u(500,'(Btu/h)/ton_of_refrigeration')
  else:
    power_per_capacity = fr_u(283,'(Btu/h)/ton_of_refrigeration')
  return power_per_capacity/flow_per_capacity

def title24_gross_total_cooling_capacity(conditions, system):
  shr = title24_shr(conditions)
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Title 24 curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  T_idb = to_u(conditions.indoor.db,"°F") # Title 24 curves use °F
  CFM_per_ton = to_u(conditions.std_air_vol_flow_per_capacity,"cu_ft/min/ton_of_refrigeration")
  if shr < 1:
    coeffs = [0,0.009645900,0.002536900,0.000171500,0,0,-0.000095900,0.000008180,-0.000007550,0.000105700,-53.542300000,0.381567150]
  else: # shr == 1
    coeffs = [0.009483100,0,-0.000600600,-0.000148900,-0.000032600,0.000011900,0,0,-0.000005050,0,-52.561740000,0.430751600]
  return CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

def title24_gross_sensible_cooling_capacity(conditions, system):
  return title24_shr(conditions)*system.gross_total_cooling_capacity(conditions)

def title24_gross_cooling_power(conditions, system):
  shr = title24_shr(conditions)
  T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Title 24 curves use °F
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  T_idb = to_u(conditions.indoor.db,"°F") # Title 24 curves use °F
  CFM_per_ton = to_u(conditions.std_air_vol_flow_per_capacity,"cu_ft/min/ton_of_refrigeration")
  cap95 = system.net_total_cooling_capacity_rated[conditions.compressor_speed]
  q_fan = system.cooling_fan_power_rated[conditions.compressor_speed]
  if T_odb < 95.0:
    seer = fr_u(system.kwargs["input_seer"],'Btu/Wh')
    if shr < 1:
      seer_coeffs = [0,-0.0202256,0.0236703,-0.0006638,0,0,-0.0001841,0.0000214,-0.00000812,0.0002971,-27.95672,0.209951063]
      cap_coeffs = [0,0.009645900,0.002536900,0.000171500,0,0,-0.000095900,0.000008180,-0.000007550,0.000105700,-53.542300000,0.381567150]
    else: # shr == 1
      seer_coeffs = [0.0046103,0,0.0125598,-0.000512,-0.0000357,0.0000105,0,0,0,0,0,-0.316172311]
      cap_coeffs = [0.009483100,0,-0.000600600,-0.000148900,-0.000032600,0.000011900,0,0,-0.000005050,0,-52.561740000,0.430751600]
    f_cond_seer = CA_regression(cap_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)/CA_regression(seer_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
    seer_nf = f_cond_seer*(1.09*cap95+q_fan)/(1.09*cap95/seer - q_fan) # unitless
  else:
    seer_nf = 0.0
  if T_odb > 82.0:
    eer = system.net_cooling_cop_rated[conditions.compressor_speed]
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

def title24_cap17_ratio_rated(hspf):
  '''
  Return the ratio of net integrated heating capacity for 47 F / 17 F.
  '''
  if hspf < 7.5:
    return 0.1113 * hspf - 0.22269
  elif hspf < 9.5567:
    return 0.017 * hspf + 0.4804
  elif hspf < 10.408:
    return 0.0982 * hspf - 0.2956
  else:
    return 0.0232 * hspf + 0.485

def title24_get_cap17(conditions, system):
  '''
  Return the net integrated heating capacity at 17 F.
  '''
  # If not already in the model data, initialize the model data
  if "cap17" not in system.model_data:
    system.model_data["cap17"] = [None]*system.number_of_speeds

  if system.model_data["cap17"][conditions.compressor_speed] is not None:
    # If it's already in the model data, return the stored value
    return system.model_data["cap17"][conditions.compressor_speed]
  else:
    # If not already in the model data then...
    if "cap17" in system.kwargs:
      # Read from model kwargs (if provided)
      system.model_data["cap17"][conditions.compressor_speed] = system.kwargs["cap17"][conditions.compressor_speed]
    else:
      # or use the Title 24 default calculation
      cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
      system.model_data["cap17"][conditions.compressor_speed] = title24_cap17_ratio_rated(system.kwargs["input_hspf"])*cap47
    return system.model_data["cap17"][conditions.compressor_speed]

def title24_get_cap35(conditions, system):
  if "cap35" not in system.model_data:
    system.model_data["cap35"] = [None]*system.number_of_speeds

  if system.model_data["cap35"][conditions.compressor_speed] is not None:
    return system.model_data["cap35"][conditions.compressor_speed]
  else:
    if "cap35" in system.kwargs:
      system.model_data["cap35"][conditions.compressor_speed] = system.kwargs["cap35"][conditions.compressor_speed]
    else:
      cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
      cap17 = title24_get_cap17(conditions, system)
      cap35 = cap17 + 0.6*(cap47 - cap17)
      if system.defrost.strategy != DefrostStrategy.NONE:
        cap35 *= 0.9
      system.model_data["cap35"][conditions.compressor_speed] = cap35
    return system.model_data["cap35"][conditions.compressor_speed]

def title24_gross_steady_state_heating_capacity(conditions, system):
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
  cap17 = title24_get_cap17(conditions, system)
  slope = (cap47 - cap17)/(47.0 - 17.0)
  return cap17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

def title24_gross_integrated_heating_capacity(conditions, system):
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
  cap17 = title24_get_cap17(conditions, system)
  cap35 = title24_get_cap35(conditions, system)
  if system.defrost.in_defrost(conditions) and (T_odb > 17.0 and T_odb < 45.0):
    slope = (cap35 - cap17)/(35.0 - 17.0)
  else:
    slope = (cap47 - cap17)/(47.0 - 17.0)
  return cap17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

def title24_net_heating_cop_rated(hspf):
  return 0.3225*hspf + 0.9099

def title24_check_hspf(conditions, system, cop17):
  # Calculate region 4 HSPF
  cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
  cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
  inp47 = cap47/cop47
  cap35 = title24_get_cap35(conditions, system)
  cap17 = title24_get_cap17(conditions, system)
  inp17 = cap17/cop17

  if "cop35" in system.kwargs:
    cop35 = system.kwargs["cop35"][conditions.compressor_speed]
    system.model_data["cop35"][conditions.compressor_speed] = cop35
    inp35 = cap35/cop35
  else:
    inp35 = inp17 + 0.6*(inp47 - inp17)
    if system.defrost.strategy != DefrostStrategy.NONE:
      inp35 *= 0.985
    cop35 = cap35/inp35
    system.model_data["cop35"][conditions.compressor_speed] = cop35

  out_tot = 0
  inp_tot = 0

  T_bins = [62.0, 57.0, 52.0, 47.0, 42.0, 37.0, 32.0, 27.0, 22.0, 17.0, 12.0, 7.0, 2.0, -3.0, -8.0]
  frac_hours = [0.132, 0.111, 0.103, 0.093, 0.100, 0.109, 0.126, 0.087, 0.055, 0.036, 0.026, 0.013, 0.006, 0.002, 0.001]

  T_design = 5.0
  T_edb = 65.0
  C = 0.77  # AHRI "correction factor"
  T_off = 0.0  # low temp cut-out "off" temp (F)
  T_on = 5.0  # low temp cut-out "on" temp (F)
  dHRmin = cap47

  for i, T_odb in enumerate(T_bins):
    bL = ((T_edb - T_odb) / (T_edb - T_design)) * C * dHRmin

    if (T_odb > 17.0 and T_odb < 45.0):
      cap_slope = (cap35 - cap17)/(35.0 - 17.0)
      inp_slope = (inp35 - inp17)/(35.0 - 17.0)
    else:
      cap_slope = (cap47 - cap17)/(47.0 - 17.0)
      inp_slope = (inp47 - inp17)/(47.0 - 17.0)
    cap = cap17 + cap_slope*(T_odb - 17.0)
    inp = inp17 + inp_slope*(T_odb - 17.0)

    x_t = min(bL/cap, 1.0)
    PLF = 1.0 - (system.c_d_heating * (1.0 - x_t))
    if T_odb <= T_off or cap/inp < 1.0:
      sigma_t = 0.0
    elif T_odb <= T_on:
      sigma_t = 0.5
    else:
      sigma_t = 1.0

    inp_tot += x_t*inp*sigma_t/PLF*frac_hours[i] + (bL - (x_t*cap*sigma_t))*frac_hours[i]
    out_tot += bL*frac_hours[i]

  return to_u(out_tot/inp_tot,"Btu/Wh")

def title24_cop47_rated(hspf):
  return 0.3225*hspf + 0.9099

def title24_c_d_heating(hspf):
  return max(min(.25 - 0.2*(hspf-6.8)/(10.0-6.8),0.25),0.05)

def title24_calculate_cops(conditions, system):
  if "cop35" not in system.model_data:
    system.model_data["cop35"] = [None]*system.number_of_speeds

  if "cop17" not in system.model_data:
    system.model_data["cop17"] = [None]*system.number_of_speeds

  hspf = system.kwargs["input_hspf"]
  root_fn = lambda cop17 : title24_check_hspf(conditions, system, cop17) - hspf
  cop17_guess = 3.0 #0.2186*hspf + 0.6734
  system.model_data["cop17"][conditions.compressor_speed] = optimize.newton(root_fn, cop17_guess)

def title24_get_cop35(conditions, system):
  if "cop35" not in system.model_data:
    title24_calculate_cops(conditions, system)

  return system.model_data["cop35"][conditions.compressor_speed]

def title24_get_cop17(conditions, system):
  if "cop17" not in system.model_data:
    title24_calculate_cops(conditions, system)

  return system.model_data["cop17"][conditions.compressor_speed]

def title24_gross_steady_state_heating_power(conditions, system):
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
  cap17 = title24_get_cap17(conditions, system)

  cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
  cop17 = title24_get_cop17(conditions, system)

  inp47 = cap47/cop47
  inp17 = cap17/cop17

  slope = (inp47 - inp17)/(47.0 - 17.0)
  return inp17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

def title24_gross_integrated_heating_power(conditions, system):
  T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
  cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
  cap35 = title24_get_cap35(conditions, system)
  cap17 = title24_get_cap17(conditions, system)

  cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
  cop35 = title24_get_cop35(conditions, system)
  cop17 = title24_get_cop17(conditions, system)

  inp47 = cap47/cop47
  inp35 = cap35/cop35
  inp17 = cap17/cop17

  if system.defrost.in_defrost(conditions) and (T_odb > 17.0 and T_odb < 45.0):
    slope = (inp35 - inp17)/(35.0 - 17.0)
  else:
    slope = (inp47 - inp17)/(47.0 - 17.0)
  return inp17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

# Unified RESNET Model
resnet_gross_cooling_power = cutler_gross_cooling_power
resnet_gross_total_cooling_capacity = cutler_gross_total_cooling_capacity
resnet_gross_sensible_cooling_capacity = energyplus_gross_sensible_cooling_capacity
resnet_shr_rated = title24_shr
resnet_gross_steady_state_heating_capacity = cutler_gross_steady_state_heating_capacity
resnet_gross_integrated_heating_capacity = epri_gross_integrated_heating_capacity
resnet_gross_steady_state_heating_power = cutler_gross_steady_state_heating_power
resnet_gross_integrated_heating_power = epri_gross_integrated_heating_power
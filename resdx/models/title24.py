from enum import Enum
from scipy import optimize

from ..units import fr_u, to_u
from ..defrost import DefrostStrategy

from .base_model import DXModel

class Title24DXModel(DXModel):

  @staticmethod
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

  @staticmethod
  def gross_shr(conditions):
    T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Cutler curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    T_idb = to_u(conditions.indoor.db,"°F") # Title 24 curves use °F
    CFM_per_ton = to_u(conditions.std_air_vol_flow_per_capacity,"cu_ft/min/ton_of_refrigeration")
    coeffs = [0.0242020,-0.0592153,0.0012651,0.0016375,0,0,0,-0.0000165,0,0.0002021,0,1.5085285]
    SHR = Title24DXModel.CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
    return min(1.0, SHR)

  @staticmethod
  def eer_rated(seer):
    if seer < 13.0:
      return 10.0  + 0.84 * (seer - 11.5)
    elif seer < 16.0:
      return 11.3 + 0.57 * (seer - 13.0)
    else:
      return 13.0

  class MotorType(Enum):
    PSC = 1,
    BPM = 2

  @staticmethod
  def fan_efficacy_rated(flow_per_capacity, motor_type=MotorType.PSC):
    if motor_type == Title24DXModel.MotorType.PSC:
      power_per_capacity = fr_u(500,'(Btu/h)/ton_of_refrigeration')
    else:
      power_per_capacity = fr_u(283,'(Btu/h)/ton_of_refrigeration')
    return power_per_capacity/flow_per_capacity

  @staticmethod
  def gross_total_cooling_capacity(conditions, system):
    shr = Title24DXModel.gross_shr(conditions)
    T_iwb = to_u(conditions.indoor.get_wb(),"°F") # Title 24 curves use °F
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    T_idb = to_u(conditions.indoor.db,"°F") # Title 24 curves use °F
    CFM_per_ton = to_u(conditions.std_air_vol_flow_per_capacity,"cu_ft/min/ton_of_refrigeration")
    if shr < 1:
      coeffs = [0,0.009645900,0.002536900,0.000171500,0,0,-0.000095900,0.000008180,-0.000007550,0.000105700,-53.542300000,0.381567150]
    else: # shr == 1
      coeffs = [0.009483100,0,-0.000600600,-0.000148900,-0.000032600,0.000011900,0,0,-0.000005050,0,-52.561740000,0.430751600]
    return Title24DXModel.CA_regression(coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)*system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

  @staticmethod
  def gross_sensible_cooling_capacity(conditions, system):
    return Title24DXModel.gross_shr(conditions)*system.gross_total_cooling_capacity(conditions)

  @staticmethod
  def gross_cooling_power(conditions, system):
    shr = Title24DXModel.gross_shr(conditions)
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
      f_cond_seer = Title24DXModel.CA_regression(cap_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)/Title24DXModel.CA_regression(seer_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
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
      f_cond_eer = Title24DXModel.CA_regression(eer_coeffs,T_iwb,T_odb,T_idb,CFM_per_ton)
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

  @staticmethod
  def cap17_ratio_rated(hspf):
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

  @staticmethod
  def get_cap17(conditions, system):
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
        system.model_data["cap17"][conditions.compressor_speed] = Title24DXModel.cap17_ratio_rated(system.kwargs["input_hspf"])*cap47
      return system.model_data["cap17"][conditions.compressor_speed]

  @staticmethod
  def get_cap35(conditions, system):
    if "cap35" not in system.model_data:
      system.model_data["cap35"] = [None]*system.number_of_speeds

    if system.model_data["cap35"][conditions.compressor_speed] is not None:
      return system.model_data["cap35"][conditions.compressor_speed]
    else:
      if "cap35" in system.kwargs:
        system.model_data["cap35"][conditions.compressor_speed] = system.kwargs["cap35"][conditions.compressor_speed]
      else:
        cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
        cap17 = Title24DXModel.get_cap17(conditions, system)
        cap35 = cap17 + 0.6*(cap47 - cap17)
        if system.defrost.strategy != DefrostStrategy.NONE:
          cap35 *= 0.9
        system.model_data["cap35"][conditions.compressor_speed] = cap35
      return system.model_data["cap35"][conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_capacity(conditions, system):
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
    cap17 = Title24DXModel.get_cap17(conditions, system)
    slope = (cap47 - cap17)/(47.0 - 17.0)
    return cap17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
    cap17 = Title24DXModel.get_cap17(conditions, system)
    cap35 = Title24DXModel.get_cap35(conditions, system)
    if system.defrost.in_defrost(conditions) and (T_odb > 17.0 and T_odb < 45.0):
      slope = (cap35 - cap17)/(35.0 - 17.0)
    else:
      slope = (cap47 - cap17)/(47.0 - 17.0)
    return cap17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

  @staticmethod
  def net_heating_cop_rated(hspf):
    return 0.3225*hspf + 0.9099

  @staticmethod
  def check_hspf(conditions, system, cop17):
    # Calculate region 4 HSPF
    cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
    cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
    inp47 = cap47/cop47
    cap35 = Title24DXModel.get_cap35(conditions, system)
    cap17 = Title24DXModel.get_cap17(conditions, system)
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

  @staticmethod
  def cop47_rated(hspf):
    return 0.3225*hspf + 0.9099

  @staticmethod
  def c_d_heating(hspf):
    return max(min(.25 - 0.2*(hspf-6.8)/(10.0-6.8),0.25),0.05)

  @staticmethod
  def calculate_cops(conditions, system):
    if "cop35" not in system.model_data:
      system.model_data["cop35"] = [None]*system.number_of_speeds

    if "cop17" not in system.model_data:
      system.model_data["cop17"] = [None]*system.number_of_speeds

    hspf = system.kwargs["input_hspf"]
    root_fn = lambda cop17 : Title24DXModel.check_hspf(conditions, system, cop17) - hspf
    cop17_guess = 3.0 #0.2186*hspf + 0.6734
    system.model_data["cop17"][conditions.compressor_speed] = optimize.newton(root_fn, cop17_guess)

  @staticmethod
  def get_cop35(conditions, system):
    if "cop35" not in system.model_data:
      Title24DXModel.calculate_cops(conditions, system)

    return system.model_data["cop35"][conditions.compressor_speed]

  @staticmethod
  def get_cop17(conditions, system):
    if "cop17" not in system.model_data:
      Title24DXModel.calculate_cops(conditions, system)

    return system.model_data["cop17"][conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
    cap17 = Title24DXModel.get_cap17(conditions, system)

    cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
    cop17 = Title24DXModel.get_cop17(conditions, system)

    inp47 = cap47/cop47
    inp17 = cap17/cop17

    slope = (inp47 - inp17)/(47.0 - 17.0)
    return inp17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    T_odb = to_u(conditions.outdoor.db,"°F") # Title 24 curves use °F
    cap47 = system.net_heating_capacity_rated[conditions.compressor_speed]
    cap35 = Title24DXModel.get_cap35(conditions, system)
    cap17 = Title24DXModel.get_cap17(conditions, system)

    cop47 = system.net_heating_cop_rated[conditions.compressor_speed]
    cop35 = Title24DXModel.get_cop35(conditions, system)
    cop17 = Title24DXModel.get_cop17(conditions, system)

    inp47 = cap47/cop47
    inp35 = cap35/cop35
    inp17 = cap17/cop17

    if system.defrost.in_defrost(conditions) and (T_odb > 17.0 and T_odb < 45.0):
      slope = (inp35 - inp17)/(35.0 - 17.0)
    else:
      slope = (inp47 - inp17)/(47.0 - 17.0)
    return inp17 + slope*(T_odb - 17.0) - system.heating_fan_power_rated[conditions.compressor_speed]
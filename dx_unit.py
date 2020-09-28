
#%%
import numpy as np
import sys
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
                    indoor_drybulb=u(80.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=1.0, # operating flow/ rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction  # Still need to figure out how to actually use this
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)

class HeatingConditions:
  def __init__(self,outdoor_drybulb=u(47.0,"°F"),
                    indoor_rh=0.4,
                    outdoor_rh=0.4,
                    indoor_drybulb=u(70.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=1.0, # operating flow/ rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction  # Still need to figure out how to actually use this
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)
    self.outdoor_rh = outdoor_rh

class DXUnit:

  A_cond = CoolingConditions()
  B_cond = CoolingConditions(outdoor_drybulb=u(82.0,"°F"))
  F_cond = CoolingConditions(outdoor_drybulb=u(67.0,"°F"),compressor_speed=1)
  H1_full_cond = HeatingConditions()
  H3_full_cond = HeatingConditions(outdoor_drybulb=u(17.0,"°F"))
  H2_full_cond = HeatingConditions(outdoor_drybulb=u(35.0,"°F"))
  H0_low_cond = HeatingConditions(outdoor_drybulb=u(62.0,"°F"),compressor_speed=1)
  H1_low_cond = HeatingConditions(compressor_speed=1)
  H3_low_cond = HeatingConditions(outdoor_drybulb=u(17.0,"°F"),compressor_speed=1)
  H2_low_cond = HeatingConditions(outdoor_drybulb=u(35.0,"°F"),compressor_speed=1)

  def __init__(self,gross_total_cooling_capacity=lambda conditions, scalar : 10000,
                    gross_sensible_cooling_capacity=lambda conditions : 1.0,
                    gross_cooling_power=lambda conditions, scalar : 8000,
                    c_d_cooling=0.2,
                    fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')],
                    cop_cooling_rated=[3.0],
                    flow_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_cooling_rated=[11000], # This has same units as gross capacity. This should be in base units to remove confusion.
                    shr_cooling_rated=[0.8], # Sensible heat ratio (Sensible capacity / Total capacity)
                    gross_stead_state_heating_capacity=lambda conditions, scalar : 10000, # in base unit to remove confusion
                    gross_integrated_heating_capacity=lambda conditions : 1.0,
                    gross_stead_state_heating_power=lambda conditions, scalar : 8000, # in base unit to remove confusion
                    gross_integrated_heating_power=lambda conditions : 1.0,
                    c_d_heating=0.2,
                    fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')],
                    cop_heating_rated=[2.5],
                    flow_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_heating_rated=[11000], # # This has same units as gross capacity. This should be in base units to remove confusion.
                    climate_region = 4,
                    defrost_strategy = 'demand_defrost',
                    temp_frost_influence_start = [45], # eq. 11.119 and 11.134 this is in °F
                    minimum_hp_outdoor_temp_low = 10.0, # value taken from Scott's script single-stage
                    cycling = 'between_low_full', # 'between_off_full' | 'between_low_full'
                    minimum_hp_outdoor_temp_high = 14.0): # value taken from Scott's script single-stage
    self.number_of_speeds = len(cop_cooling_rated)
    self.gross_total_cooling_capacity = gross_total_cooling_capacity # This should be in base units
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
    self.c_d_heating = c_d_heating
    self.cycling = cycling
    self.fan_eff_heating_rated = fan_eff_heating_rated
    self.cop_heating_rated = cop_heating_rated
    self.flow_per_cap_heating_rated = flow_per_cap_heating_rated
    self.cap_heating_rated = cap_heating_rated
    self.climate_region = climate_region
    self.defrost_strategy = defrost_strategy
    self.temp_frost_influence_start = temp_frost_influence_start
    self.minimum_hp_outdoor_temp_low = minimum_hp_outdoor_temp_low
    self.minimum_hp_outdoor_temp_high = minimum_hp_outdoor_temp_high
    self.regions_table_htg = {1: {'htg_load_hrs':750,'Out_design_temp':37,'fraction_hrs':[0.291,0.239,0.194,0.129,0.081,0.041,0.019,0.005,0.001,0,0,0,0,0,0,0,0,0]},
                              2: {'htg_load_hrs':1250,'Out_design_temp':27,'fraction_hrs':[0.215,0.189,0.163,0.143,0.112,0.088,0.056,0.024,0.008,0.002,0,0,0,0,0,0,0,0]},
                              3: {'htg_load_hrs':1750,'Out_design_temp':17,'fraction_hrs':[0.153,0.142,0.138,0.137,0.135,0.118,0.092,0.047,0.021,0.009,0.005,0.002,0.001,0,0,0,0,0]},
                              4: {'htg_load_hrs':2250,'Out_design_temp':5,'fraction_hrs':[0.132,0.111,0.103,0.093,0.100,0.109,0.126,0.087,0.055,0.036,0.026,0.013,0.006,0.002,0.001,0,0,0]},
                              5: {'htg_load_hrs':2750,'Out_design_temp':-10,'fraction_hrs':[0.106,0.092,0.086,0.076,0.078,0.087,0.102,0.094,0.074,0.055,0.047,0.038,0.029,0.018,0.010,0.005,0.002,0.001]},
                              6: {'htg_load_hrs':2750,'Out_design_temp':30,'fraction_hrs':[0.113,0.206,0.215,0.204,0.141,0.076,0.034,0.008,0.003,0,0,0,0,0,0,0,0,0]},
                              'bin':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                              'Temp':[62,57,52,47,42,37,32,27,22,17,12,7,2,-3,-8,-13,-18,-23]
                             }
    self.table_cooling = {'bin':[1,2,3,4,5,6,7,8],
                          'Temp': [67,72,77,82,87,92,97,102],
                          'fraction_hrs': [0.214,0.231,0.216,0.161,0.104,0.052,0.018,0.004]
                         }
    self.standar_design_htg_requirements = [(5000+i*5000)/3.412 for i in range(0,8)] + [(50000+i*10000)/3.412 for i in range(0,9)] # division by 3.412 is used for btu/h to w conversion

    # Check to make sure all cooling arrays are the same size if not output a warning message
    self.check_array_lengths()

    # Check to make sure arrays are in descending order
    self.check_array_order(self.cap_cooling_rated)
    self.check_array_order(self.cap_heating_rated)

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
    self.check_array_length(self.temp_frost_influence_start)

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

  def building_load_cooling(self):
    sizing_factor = 1.1 # eq. 11.61
    BL = np.asarray([(u(self.table_cooling['Temp'][i],"°F")-u(65.0,"°F"))/(u(95,"°F")-u(65.0,"°F")) for i in range(0,8)]) * self.net_total_cooling_capacity(self.A_cond) / sizing_factor # eq. 11.60
    return BL # BL is an array with a length equal to # of bins

  def cooling_capacity_power_bins(self):
    q_full_bin = {'cap': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    q_low_bin = {'cap': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    p_full_bin = {'power': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    p_low_bin = {'power': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    for t_bin in self.table_cooling['Temp']:
        q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] = self.net_total_cooling_capacity(self.B_cond) + (self.net_total_cooling_capacity(self.A_cond) - self.net_total_cooling_capacity(self.B_cond)) * ((u(t_bin,"°F")-self.B_cond.outdoor_drybulb)/(self.A_cond.outdoor_drybulb-self.B_cond.outdoor_drybulb)) # e.q. 11.64
        q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] = self.net_total_cooling_capacity(self.F_cond) + (self.net_total_cooling_capacity(self.B_cond) - self.net_total_cooling_capacity(self.F_cond)) * ((u(t_bin,"°F")-self.F_cond.outdoor_drybulb)/(self.B_cond.outdoor_drybulb-self.F_cond.outdoor_drybulb)) # e.q. 11.62
        p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)] = self.net_cooling_power(self.B_cond) + (self.net_cooling_power(self.A_cond) - self.net_cooling_power(self.B_cond)) * ((u(t_bin,"°F")-self.B_cond.outdoor_drybulb)/(self.A_cond.outdoor_drybulb-self.B_cond.outdoor_drybulb)) # e.q. 11.65
        p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)] = self.net_cooling_power(self.F_cond) + (self.net_cooling_power(self.B_cond) - self.net_cooling_power(self.F_cond)) * ((u(t_bin,"°F")-self.F_cond.outdoor_drybulb)/(self.B_cond.outdoor_drybulb-self.F_cond.outdoor_drybulb)) # e.q. 11.63
    return q_full_bin, q_low_bin, p_full_bin, p_low_bin

  def total_bin_capacity_cooling(self):
    q = {'cap': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    bl = {'bl': self.building_load_cooling(),'t_bin': self.table_cooling['Temp']}
    q_full_bin, q_low_bin,_,_ = self.cooling_capacity_power_bins()
    clf_full_bin, clf_low_bin = self.cooling_load_factor()
    hrs_fraction = {'fraction': self.table_cooling['fraction_hrs'],'t_bin': self.table_cooling['Temp']}
    for t_bin in self.table_cooling['Temp']:
        if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q. 11.66
            q['cap'][q['t_bin'].index(t_bin)] = clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)]*q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'): # e.q. 11.72
            q['cap'][q['t_bin'].index(t_bin)] = (clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)]*q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)]+clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)]*q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'): # e.q. 11.76
             q['cap'][q['t_bin'].index(t_bin)] = clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)]*q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
        else: # e.q. 11.81
            q['cap'][q['t_bin'].index(t_bin)] = q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
    return q

  def total_bin_energy_cooling(self):
    e = {'e': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    bl = {'bl': self.building_load_cooling(),'t_bin': self.table_cooling['Temp']}
    q_full_bin, q_low_bin,_,_ = self.cooling_capacity_power_bins()
    _,_,p_full_bin, p_low_bin = self.cooling_capacity_power_bins()
    clf_full_bin, clf_low_bin = self.cooling_load_factor()
    plf_full_bin, plf_low_bin = self.plf_cooling_bins()
    hrs_fraction = {'fraction': self.table_cooling['fraction_hrs'],'t_bin': self.table_cooling['Temp']}
    for t_bin in self.table_cooling['Temp']:
        if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q. 11.67
            e['e'][e['t_bin'].index(t_bin)] = clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)]*p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]/plf_low_bin['plf'][plf_low_bin['t_bin'].index(t_bin)]
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'): # e.q. 11.73
            e['e'][e['t_bin'].index(t_bin)] = (clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)]*p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)]+clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)]*p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'): # e.q. 11.77
            e['e'][e['t_bin'].index(t_bin)] = clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)]*p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]/plf_full_bin['plf'][plf_full_bin['t_bin'].index(t_bin)]
        else: # e.q. 11.82
            e['e'][e['t_bin'].index(t_bin)] = p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
    return e

  def cooling_load_factor(self):
    bl = {'bl': self.building_load_cooling(),'t_bin': self.table_cooling['Temp']}
    clf_full_bin = {'clf': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    clf_low_bin = {'clf': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
    q_full_bin, q_low_bin,_,_ = self.cooling_capacity_power_bins()
    hrs_fraction = {'fraction': self.table_cooling['fraction_hrs'],'t_bin': self.table_cooling['Temp']}
    for t_bin in self.table_cooling['Temp']:
        if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q. 11.68
            clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)] = (bl['bl'][bl['t_bin'].index(t_bin)])/(q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])
            clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)] = float('nan') #use nan to indicate not defined in this conditions
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'): # e.q. 11.74
            clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)] = (q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]-bl['bl'][bl['t_bin'].index(t_bin)])/(q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]-q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])
            clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)] = 1 - clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)]
        elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'): # e.q. 11.78
            clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)] = float('nan')
            clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)] = bl['bl'][bl['t_bin'].index(t_bin)]/q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]
        else:
            clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)] = float('nan')
            clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)] = float('nan')
    return clf_full_bin, clf_low_bin

  def plf_cooling_bins(self):
    if self.number_of_speeds == 1:
        plr = 0.5 # part load ratio
        plf = 1.0 - plr*self.c_d_cooling  # eq. 11.56
        return plf
    else:
        bl = {'bl': self.building_load_cooling(),'t_bin': self.table_cooling['Temp']}
        q_full_bin, q_low_bin,_,_ = self.cooling_capacity_power_bins()
        plf_full_bin = {'plf': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
        plf_low_bin = {'plf': [i*0 for i in range(8)],'t_bin': self.table_cooling['Temp']}
        clf_full_bin, clf_low_bin = self.cooling_load_factor()
        for t_bin in self.table_cooling['Temp']:
            if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q. 11.69
                plf_low_bin['plf'][plf_low_bin['t_bin'].index(t_bin)] = 1.0- self.c_d_cooling*(1-clf_low_bin['clf'][clf_low_bin['t_bin'].index(t_bin)])
                plf_full_bin['plf'][plf_full_bin['t_bin'].index(t_bin)] = float('nan') #use nan to indicate not defined in this conditions
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'):
                plf_low_bin['plf'][plf_low_bin['t_bin'].index(t_bin)] = float('nan')
                plf_full_bin['plf'][plf_full_bin['t_bin'].index(t_bin)] = float('nan')
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'): # e.q. 11.79
                plf_low_bin['plf'][plf_low_bin['t_bin'].index(t_bin)] = float('nan')
                plf_full_bin['plf'][plf_full_bin['t_bin'].index(t_bin)] = 1.0- self.c_d_cooling*(1-clf_full_bin['clf'][clf_full_bin['t_bin'].index(t_bin)])
            else:
                plf_low_bin['plf'][plf_low_bin['t_bin'].index(t_bin)] = float('nan')
                plf_full_bin['plf'][plf_full_bin['t_bin'].index(t_bin)] = float('nan')
        return plf_full_bin, plf_low_bin

  def eer(self, conditions): # e.q. 11.17
    eer = self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)
    return eer

  def seer(self):
    if self.number_of_speeds == 1:
        seer = self.plf_cooling_bins() * self.eer(self.B_cond) # eq. 11.55
    else: # e.q. 11.59
        q = np.asarray(self.total_bin_capacity_cooling()['cap'])
        e = np.asarray(self.total_bin_energy_cooling()['e'])
        seer = np.sum(q)/np.sum(e)
    return convert(seer,'','Btu/Wh')

  ### For heating ###
  def building_load_htg(self,conditions):
    agreement_factor = 0.77 # eq. 11.110
    BL = np.asarray([(u(65.0,"°F")-u(self.regions_table_htg['Temp'][i],"°F"))/(u(65,"°F")-u(self.regions_table_htg[self.climate_region]['Out_design_temp'],"°F")) for i in range(0,18)]) * agreement_factor * self.round_to_closet_value(self.min_design_htg(conditions)) # eq. 11.109
    return BL # BL is an array with a length equal to # of bins

  def defrost_factor(self): # eq. 11.129
    t_test = 90 # this is in minutes
    t_max  = 720
    if self.defrost_strategy == 'demand_defrost':
       f_def = 1 + 0.03 * (1 - (t_test-90)/(t_max-90))
    else:
       f_def = 1
    return f_def

  def net_total_heating_capacity(self, conditions):
    return self.gross_stead_state_heating_capacity(conditions,self.cap_heating_rated[conditions.compressor_speed]) + self.fan_heat(conditions) # eq. 11.31

  def heating_capacity_bins(self):
    q_full_bin = {'cap': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']: #  eq. 11.117 and 11.118
        if (t_bin >= self.temp_frost_influence_start[0]) or (t_bin <= 17):
            q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_full_cond) + (self.net_total_heating_capacity(self.H1_full_cond) - self.net_total_heating_capacity(self.H3_full_cond)) * ((u(t_bin,"°F")-self.H3_full_cond.outdoor_drybulb)/(self.H1_full_cond.outdoor_drybulb-self.H3_full_cond.outdoor_drybulb))
        else:
            q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_full_cond) + (self.net_total_heating_capacity(self.H2_full_cond) - self.net_total_heating_capacity(self.H3_full_cond)) * ((u(t_bin,"°F")-self.H3_full_cond.outdoor_drybulb)/(self.H2_full_cond.outdoor_drybulb-self.H3_full_cond.outdoor_drybulb))
    if self.number_of_speeds == 2: # double-stage system eq. 11.135 to 11.137
        q_low_bin = {'cap': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        for t_bin in self.regions_table_htg['Temp']:
            if (t_bin >= self.temp_frost_influence_start[1]): # temp_frost_influence_start[1] temperature for 2nd stage eq. 11.134
                q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H1_low_cond) + (self.net_total_heating_capacity(self.H0_low_cond) - self.net_total_heating_capacity(self.H1_low_cond)) * ((u(t_bin,"°F")-self.H1_low_cond.outdoor_drybulb)/(self.H0_low_cond.outdoor_drybulb-self.H1_low_cond.outdoor_drybulb))
            elif (t_bin > 17) and (t_bin < self.temp_frost_influence_start[1]):
                q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_low_cond) + (self.net_total_heating_capacity(self.H2_low_cond) - self.net_total_heating_capacity(self.H3_low_cond)) * ((u(t_bin,"°F")-self.H3_low_cond.outdoor_drybulb)/(self.H2_low_cond.outdoor_drybulb-self.H3_low_cond.outdoor_drybulb))
            else:
                q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_low_cond) + (self.net_total_heating_capacity(self.H1_low_cond) - self.net_total_heating_capacity(self.H3_low_cond)) * ((u(t_bin,"°F")-self.H3_low_cond.outdoor_drybulb)/(self.H1_low_cond.outdoor_drybulb-self.H3_low_cond.outdoor_drybulb))
        return q_full_bin, q_low_bin
    else:
        return q_full_bin

  def heating_power_bins(self): # eq. 11.123 and 11.124
    p_full_bin = {'power': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']:
        if (t_bin >= self.temp_frost_influence_start[0]) or (t_bin <= 17):
            p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_full_cond) + (self.net_heating_power(self.H1_full_cond) - self.net_heating_power(self.H3_full_cond)) * ((u(t_bin,"°F")-self.H3_full_cond.outdoor_drybulb)/(self.H1_full_cond.outdoor_drybulb-self.H3_full_cond.outdoor_drybulb))
        else:
            p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_full_cond) + (self.net_heating_power(self.H2_full_cond) - self.net_heating_power(self.H3_full_cond)) * ((u(t_bin,"°F")-self.H3_full_cond.outdoor_drybulb)/(self.H2_full_cond.outdoor_drybulb-self.H3_full_cond.outdoor_drybulb))
    if self.number_of_speeds == 2: # double-stage system e.q 11.138 to 11.140
        p_low_bin = {'power': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        for t_bin in self.regions_table_htg['Temp']:
            if (t_bin >= self.temp_frost_influence_start[1]): # temp_frost_influence_start[1] temperature for 2nd stage eq. 11.134
                p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H1_low_cond) + (self.net_heating_power(self.H0_low_cond) - self.net_heating_power(self.H1_low_cond)) * ((u(t_bin,"°F")-self.H1_low_cond.outdoor_drybulb)/(self.H0_low_cond.outdoor_drybulb-self.H1_low_cond.outdoor_drybulb))
            elif (t_bin > 17) and (t_bin < self.temp_frost_influence_start[1]):
                p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_low_cond) + (self.net_heating_power(self.H2_low_cond) - self.net_heating_power(self.H3_low_cond)) * ((u(t_bin,"°F")-self.H3_low_cond.outdoor_drybulb)/(self.H2_low_cond.outdoor_drybulb-self.H3_low_cond.outdoor_drybulb))
            else:
                p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_low_cond) + (self.net_heating_power(self.H1_low_cond) - self.net_heating_power(self.H3_low_cond)) * ((u(t_bin,"°F")-self.H3_low_cond.outdoor_drybulb)/(self.H1_low_cond.outdoor_drybulb-self.H3_low_cond.outdoor_drybulb))
        return p_full_bin, p_low_bin
    else:
        return p_full_bin

  def hp_low_temp_cutout_factor(self):
    delta_bin = {'delta': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']} # delta_bin = delta_bin_2 eq. 11.120 to 11.122 or 11.159 to 11.161
    if self.number_of_speeds == 1:
        q_full_bin = self.heating_capacity_bins()
        p_full_bin = self.heating_power_bins()
    else:
        q_full_bin = self.heating_capacity_bins()[0]
        p_full_bin = self.heating_power_bins()[0]

    for t_bin in self.regions_table_htg['Temp']:
        if ((t_bin <= self.minimum_hp_outdoor_temp_low) or (((q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)])/(p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)])) < 1)):
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 0
        elif ((t_bin <= self.minimum_hp_outdoor_temp_high) and (t_bin > self.minimum_hp_outdoor_temp_low) and (((q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)])/(p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)])) >= 1)) :
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 0.5
        else:
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 1
    if self.number_of_speeds == 2: # double-stage system 11.147 to 11.149
        delta_bin_1 = {'delta': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        q_low_bin = self.heating_capacity_bins()[1]
        p_low_bin = self.heating_power_bins()[1]
        for t_bin in self.regions_table_htg['Temp']:
            if ((t_bin <= self.minimum_hp_outdoor_temp_low) or (((q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])/(p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)])) < 1)):
                delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)] = 0
            elif ((t_bin <= self.minimum_hp_outdoor_temp_high) and (t_bin > self.minimum_hp_outdoor_temp_low)): # and (((q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])/(p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)])) >= 1)) : (This is probably wrong!)
                delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)] = 0.5
            else:
                delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)] = 1
        return delta_bin, delta_bin_1
    else:
        return delta_bin

  def net_heating_power(self, conditions):
    return self.gross_stead_state_heating_power(conditions,self.cap_heating_rated[conditions.compressor_speed]/self.cop_heating_rated[conditions.compressor_speed]) - self.fan_power(conditions) # eq. 11.41

  # def max_design_htg(self,conditions): # This won't be used for now according to note under Table 17 in AHRI standard 2017
  #   if self.climate_region == 5:
  #       dhr_max = 2.2 * self.net_total_heating_capacity(conditions)
  #   else:
  #       dhr_max = 2 * self.net_total_heating_capacity(conditions) *(u(65,"°F")-u(self.regions_table_htg[self.climate_region]['Out_design_temp'],"°F"))/(u(60,"°R"))
  #   return dhr_max

  def min_design_htg(self,conditions): # eq. 11.111 to 11.114. The equations seem to be incorrect.
    # if self.climate_region == 5:
    #     dhr_min = self.net_total_heating_capacity(conditions)
    # else:
    return self.net_total_heating_capacity(conditions) *(u(65,"°F")-u(self.regions_table_htg[self.climate_region]['Out_design_temp'],"°F"))/(u(60,"°R"))

  def round_to_closet_value(self,dhr):
    standar_design_htg_requirements = np.asarray(self.standar_design_htg_requirements)
    index = (np.abs(standar_design_htg_requirements - dhr)).argmin()
    return standar_design_htg_requirements[index]

  def hp_htg_load_factor(self): # eq. 11.115 and 11.116
    bl = {'bl':self.building_load_htg(self.H1_full_cond),'t_bin': self.regions_table_htg['Temp']}
    hlf_full_bin = {'hlf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    if self.number_of_speeds == 1:
        q_full_bin = self.heating_capacity_bins()
        for t_bin in self.regions_table_htg['Temp']:
            if q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] > bl['bl'][bl['t_bin'].index(t_bin)]:
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = (bl['bl'][bl['t_bin'].index(t_bin)])/(q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)])
            else:
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = 1
        return hlf_full_bin
    else: # double-stage system
        hlf_low_bin = {'hlf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        q_full_bin, q_low_bin = self.heating_capacity_bins()
        for t_bin in self.regions_table_htg['Temp']:
            if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q 11.143
                hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)] = (bl['bl'][bl['t_bin'].index(t_bin)])/(q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = float('nan') #use nan to indicate not defined in this conditions
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'): # e.q 11.151 and # e.q 11.152
                hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)] = (q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]-bl['bl'][bl['t_bin'].index(t_bin)])/(q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]-q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)])
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = 1 - hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)]
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'):  # e.q 11.154 and # e.q 11.155
                hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)] = float('nan')
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = bl['bl'][bl['t_bin'].index(t_bin)]/q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]
            else:  # e.q 11.158
                hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)] = float('nan')
                hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)] = 1
        return hlf_full_bin, hlf_low_bin

  def plf_htg(self): # eq. 11.125
    if self.number_of_speeds == 1:
        hlf_full_bin = self.hp_htg_load_factor()['hlf']
        hlf = np.asarray(hlf_full_bin)
        plf_full = 1 - self.c_d_heating * (1 - hlf)
        plf_full = {'plf':plf_full,'t_bin': self.regions_table_htg['Temp']}
        return plf_full
    else: # double-stage system
        plf_low = {'plf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        plf_full = {'plf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        q_full_bin,q_low_bin= self.heating_capacity_bins()
        bl = {'bl':self.building_load_htg(self.H1_full_cond),'t_bin': self.regions_table_htg['Temp']}
        hlf_full_bin, hlf_low_bin = self.hp_htg_load_factor()
        for t_bin in self.regions_table_htg['Temp']:
            if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]:  # e.q 11.144
                plf_low['plf'][plf_low['t_bin'].index(t_bin)] = 1 - self.c_d_heating * (1 - hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)])
                plf_full['plf'][plf_full['t_bin'].index(t_bin)] = float('nan')
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'):
                plf_low['plf'][plf_low['t_bin'].index(t_bin)] = float('nan')
                plf_full['plf'][plf_full['t_bin'].index(t_bin)] = float('nan')
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'):  # e.q 11.155
                plf_low['plf'][plf_low['t_bin'].index(t_bin)] = float('nan')
                plf_full['plf'][plf_full['t_bin'].index(t_bin)] = 1 - self.c_d_heating * (1 - hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)])
            else:
                plf_low['plf'][plf_low['t_bin'].index(t_bin)] = float('nan')
                plf_full['plf'][plf_full['t_bin'].index(t_bin)] = float('nan')
        return plf_full, plf_low


  def resistance_heat(self): # eq. 11.126
    bl = self.building_load_htg(self.H1_full_cond)
    if self.number_of_speeds == 1:
        hlf_full_bin = self.hp_htg_load_factor()
        hlf = np.asarray(hlf_full_bin['hlf'])
        q_full_bin = self.heating_capacity_bins()
        q_full_bin = np.asarray(q_full_bin['cap'])
        delta_bin = self.hp_low_temp_cutout_factor()
        delta_bin = np.asarray(delta_bin['delta'])
        hrs_fraction = np.asarray(self.regions_table_htg[self.climate_region]['fraction_hrs'])
        rh = {'rh': ((bl-((q_full_bin*hlf)*delta_bin)))*hrs_fraction,'t_bin': self.regions_table_htg['Temp']}
    else: # double-stage system
        rh = {'rh': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        plf_low = {'plf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
        q_full_bin,q_low_bin= self.heating_capacity_bins()
        hlf_full_bin,hlf_low_bin= self.hp_htg_load_factor()
        delta_bin, delta_bin_1 = self.hp_low_temp_cutout_factor()
        bl = {'bl':bl,'t_bin': self.regions_table_htg['Temp']}
        hrs_fraction = {'fraction': self.regions_table_htg[1]['fraction_hrs'],'t_bin': self.regions_table_htg['Temp']}
        for t_bin in self.regions_table_htg['Temp']:
            if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]: # e.q 11.143
                rh['rh'][rh['t_bin'].index(t_bin)] = bl['bl'][bl['t_bin'].index(t_bin)]*(1-delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'): # e.q 11.151 and # e.q 11.152
                rh['rh'][rh['t_bin'].index(t_bin)] = bl['bl'][bl['t_bin'].index(t_bin)]*(1-delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'):  # e.q 11.154 and # e.q 11.155
                rh['rh'][rh['t_bin'].index(t_bin)] = bl['bl'][bl['t_bin'].index(t_bin)]*(1-delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
            else:  # e.q 11.158
                rh['rh'][rh['t_bin'].index(t_bin)] = (bl['bl'][bl['t_bin'].index(t_bin)]-q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]*hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)]*delta_bin['delta'][delta_bin['t_bin'].index(t_bin)])*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
    return rh

  def bin_energy(self): # eq. 11.153 and 11.156
    e = {'e': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    bl = {'bl':self.building_load_htg(self.H1_full_cond),'t_bin': self.regions_table_htg['Temp']}
    hrs_fraction = {'fraction': self.regions_table_htg[1]['fraction_hrs'],'t_bin': self.regions_table_htg['Temp']}
    if self.number_of_speeds == 1:
        hlf_full_bin = self.hp_htg_load_factor()
        q_full_bin = self.heating_capacity_bins()
        p_full_bin = self.heating_power_bins()
        plf_full = self.plf_htg()
        delta_bin = self.hp_low_temp_cutout_factor()
        for t_bin in self.regions_table_htg['Temp']:
            e_bin = (p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)])*(hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)])*(delta_bin['delta'][delta_bin['t_bin'].index(t_bin)])*(hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)])
            if q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] > bl['bl'][bl['t_bin'].index(t_bin)]:
                e_bin = e_bin/(plf_full['plf'][plf_full['t_bin'].index(t_bin)])
            e['e'][e['t_bin'].index(t_bin)] = e_bin
    else:
        hlf_full_bin, hlf_low_bin = self.hp_htg_load_factor()
        q_full_bin,q_low_bin = self.heating_capacity_bins()
        p_full_bin,p_low_bin = self.heating_power_bins()
        plf_full, plf_low = self.plf_htg()
        delta_bin,delta_bin_1 = self.hp_low_temp_cutout_factor()
        for t_bin in self.regions_table_htg['Temp']:
            if q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] >= bl['bl'][bl['t_bin'].index(t_bin)]:  # e.q 11.141
                e['e'][e['t_bin'].index(t_bin)] = p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)]*hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)]*delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]/plf_low['plf'][plf_low['t_bin'].index(t_bin)]
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_low_full'):  # e.q 11.150
                e['e'][e['t_bin'].index(t_bin)] = (p_low_bin['power'][p_low_bin['t_bin'].index(t_bin)]*hlf_low_bin['hlf'][hlf_low_bin['t_bin'].index(t_bin)]+p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)]*hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)])*delta_bin_1['delta'][delta_bin_1['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
            elif (q_low_bin['cap'][q_low_bin['t_bin'].index(t_bin)] < bl['bl'][bl['t_bin'].index(t_bin)]) and (bl['bl'][bl['t_bin'].index(t_bin)] < q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)]) and (self.cycling == 'between_off_full'):  # e.q 11.153
                e['e'][e['t_bin'].index(t_bin)] = p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)]*hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)]*delta_bin['delta'][delta_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]/plf_full['plf'][plf_full['t_bin'].index(t_bin)]
            else:  # e.q 11.156
                e['e'][e['t_bin'].index(t_bin)] = p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)]*hlf_full_bin['hlf'][hlf_full_bin['t_bin'].index(t_bin)]*delta_bin['delta'][delta_bin['t_bin'].index(t_bin)]*hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)]
    return e

  def hspf(self): # eq. 11.108
    e =  np.asarray(self.bin_energy()['e'])
    bl = self.building_load_htg(self.H1_full_cond)
    hrs_fraction = np.asarray(self.regions_table_htg[self.climate_region]['fraction_hrs'])
    rh = np.asarray(self.resistance_heat()['rh'])
    hspf = ((np.sum(hrs_fraction * bl))/((np.sum(e)+(np.sum(rh))))) * self.defrost_factor()
    return convert(hspf,'','Btu/Wh')

  def print_cooling_info(self):
    print(f"SEER: {self.seer()}")
    for speed in range(self.number_of_speeds):
      conditions = CoolingConditions(compressor_speed=speed)
      print(f"Net cooling power for stage {speed + 1} : {self.net_cooling_power(conditions)}")
      print(f"Net cooling capacity for stage {speed + 1} : {self.net_total_cooling_capacity(conditions)}")

  def print_heating_info(self):
    print(f"HSPF: {self.hspf()}")
    for speed in range(self.number_of_speeds):
      conditions = HeatingConditions(compressor_speed=speed)
      print(f"Net heating power for stage {speed + 1} : {self.net_heating_power(conditions)}")
      print(f"Net heating capacity for stage {speed + 1} : {self.net_total_heating_capacity(conditions)}")

  def writeA205(self):
    '''Write ASHRAE 205 file!!!'''
    return

def cutler_cooling_power(conditions,scalar):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*scalar

def cutler_cooling_capacity(conditions,scalar):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*scalar

def cutler_heating_power(conditions,scalar):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*scalar

def cutler_heating_capacity(conditions,scalar):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*scalar


#%%
# Move this stuff to a separate file

# Single speed
dx_unit_1_speed = DXUnit(
  gross_total_cooling_capacity=cutler_cooling_capacity,
  gross_cooling_power=cutler_cooling_power,
  gross_stead_state_heating_capacity=cutler_heating_capacity,
  gross_stead_state_heating_power=cutler_heating_power
)

dx_unit_1_speed.print_cooling_info()

dx_unit_1_speed.print_heating_info()

# Two speed
dx_unit_2_speed = DXUnit(
  cop_cooling_rated=[3.0,3.5],
  fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')]*2,
  flow_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  cap_cooling_rated=[16000,11000],
  shr_cooling_rated=[0.8]*2,
  gross_total_cooling_capacity=cutler_cooling_capacity,
  gross_cooling_power=cutler_cooling_power,
  fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')]*2,
  cop_heating_rated=[2.5, 3.0],
  flow_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")]*2,
  cap_heating_rated=[16000,11000],
  temp_frost_influence_start=[40, 45],
  gross_stead_state_heating_capacity=cutler_heating_capacity,
  gross_stead_state_heating_power=cutler_heating_power
)

dx_unit_2_speed.print_cooling_info()

dx_unit_2_speed.print_heating_info()


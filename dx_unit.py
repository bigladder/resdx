
#%%
import numpy as np
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
                    mass_flow_fraction=0.8, # operating flow/ rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction
    self.compressor_speed = compressor_speed

class HeatingConditions:
  def __init__(self,outdoor_drybulb=u(47.0,"°F"),
                    indoor_rh=0.4,
                    outdoor_rh=0.4,
                    indoor_drybulb=u(70.0,"°F"),
                    press=u(1.0,"atm"),
                    mass_flow_fraction=0.8, # operating flow/ rated flow
                    compressor_speed=0):
    self.outdoor_drybulb = outdoor_drybulb
    self.indoor_rh = indoor_rh
    self.indoor_drybulb = indoor_drybulb
    self.press = press
    self.mass_flow_fraction=mass_flow_fraction
    self.compressor_speed = compressor_speed
    self.outdoor_rh = outdoor_rh

class DXUnit:

  A_cond = CoolingConditions()
  B_cond = CoolingConditions(outdoor_drybulb=u(82.0,"°F"))
  H1_cond = HeatingConditions()
  H3_cond = HeatingConditions(outdoor_drybulb=u(17.0,"°F"))
  H2_cond = HeatingConditions(outdoor_drybulb=u(35.0,"°F"))
  
  def __init__(self,gross_total_cooling_capacity=lambda conditions,cap_rated : 10000,
                    gross_sensible_cooling_capacity=lambda conditions : 1.0,
                    gross_cooling_power=lambda conditions, cop_rated,cap_rated : 8000,
                    c_d_cooling=0.2,
                    fan_eff_cooling_rated=[u(0.365,'W/cu_ft/min')],
                    cop_cooling_rated=[3.0],
                    flow_per_cap_cooling_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_cooling_rated=[11000], # This has same units as gross capacity. This should be in base units to remove confusion.
                    shr_cooling_rated=[0.8], # Sensible heat ratio (Sensible capacity / Total capacity)
                    gross_stead_state_heating_capacity=lambda conditions,cap_rated : 10000, # in base unit to remove confusion
                    gross_integrated_heating_capacity=lambda conditions : 1.0,
                    gross_stead_state_heating_power=lambda conditions, cop_rated,cap_rated : 8000, # in base unit to remove confusion
                    gross_integrated_heating_power=lambda conditions : 1.0,
                    c_d_heating=0.2,
                    fan_eff_heating_rated=[u(0.365,'W/cu_ft/min')],
                    cop_heating_rated=[2.5],
                    flow_per_cap_heating_rated = [u(350.0,"cu_ft/min/ton_of_refrigeration")],
                    cap_heating_rated=[11000], # # This has same units as gross capacity. This should be in base units to remove confusion.
                    climate_region = 1,
                    defrost_strategy = 'demand_defrost',
                    temp_frost_influence_start = 45, # eq. 11.119 this is in °F
                    minimum_hp_outdoor_temp_low = 10.0, # value taken from Scott's script single-stage
                    minimum_hp_outdoor_temp_high = 14.0): # value taken from Scott's script single-stage
    self.num_coolilng_speeds = len(cop_cooling_rated)
    # Check to make sure all cooling arrays are the same size if not output a warning message
    self.gross_total_cooling_capacity = gross_total_cooling_capacity # This should be in base units
    self.gross_sensible_cooling_capacity = gross_sensible_cooling_capacity
    self.gross_cooling_power = gross_cooling_power
    self.c_d_cooling = c_d_cooling
    self.fan_eff_cooling_rated = fan_eff_cooling_rated
    self.shr_cooling_rated = shr_cooling_rated
    self.cop_cooling_rated = cop_cooling_rated
    self.cap_cooling_rated = cap_cooling_rated
    self.flow_per_cap_cooling_rated = flow_per_cap_cooling_rated
    self.num_heating_speeds = len(cop_heating_rated)
    self.gross_stead_state_heating_capacity = gross_stead_state_heating_capacity
    self.gross_integrated_heating_capacity = gross_integrated_heating_capacity
    self.gross_stead_state_heating_power = gross_stead_state_heating_power
    self.gross_integrated_heating_power = gross_integrated_heating_power
    self.c_d_heating = c_d_heating
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
    self.standar_design_htg_requirements = [(5000+i*5000)/3.412 for i in range(0,8)] + [(50000+i*10000)/3.412 for i in range(0,9)] # division by 3.412 is used for btu/h to w conversion


  ### For cooling ###
  def fan_power_clg(self, conditions):
    return self.fan_eff_cooling_rated[conditions.compressor_speed]*self.standard_indoor_airflow_clg(conditions) # eq. 11.16

  def fan_heat_clg(self, conditions):
    return self.fan_power_clg(conditions)# eq. 11.11 (in SI units)

  def standard_indoor_airflow_clg(self,conditions):
    air_density_standard_conditions = 1.204 # in kg/m3
    return self.flow_per_cap_cooling_rated[conditions.compressor_speed]*self.cap_cooling_rated[conditions.compressor_speed]*psychrolib.GetDryAirDensity(convert(conditions.indoor_drybulb,"K","°C"), conditions.press)/air_density_standard_conditions

  def net_total_cooling_capacity(self, conditions):
    return self.gross_total_cooling_capacity(conditions,self.cap_cooling_rated[conditions.compressor_speed]) - self.fan_heat_clg(conditions) # eq. 11.3 but not considering duct losses

  def net_cooling_power(self, conditions):
    return self.gross_cooling_power(conditions,self.cop_cooling_rated[conditions.compressor_speed],self.cap_cooling_rated[conditions.compressor_speed]) + self.fan_power_clg(conditions) # eq. 11.15

  def eer(self, conditions):
    eer = self.net_total_cooling_capacity(conditions)/self.net_cooling_power(conditions)
    return eer

  def plf_clg(self, plr):
    return 1.0 - plr*self.c_d_cooling  # eq. 11.56

  def seer(self):
    seer = self.plf_clg(0.5) * self.eer(self.B_cond) # eq. 11.55
    return convert(seer,'','Btu/Wh')

  ### For heating ###
  def fan_power_htg(self, conditions):
    return self.fan_eff_heating_rated[conditions.compressor_speed]*self.standard_indoor_airflow_htg(conditions) # eq. 11.16

  def fan_heat_htg(self, conditions):
    return self.fan_power_htg(conditions)# eq. 11.11 (in SI units)

  def standard_indoor_airflow_htg(self,conditions):
    air_density_standard_conditions = 1.204 # in kg/m3
    return self.flow_per_cap_heating_rated[conditions.compressor_speed]*self.cap_heating_rated[conditions.compressor_speed]*psychrolib.GetDryAirDensity(convert(conditions.indoor_drybulb,"K","°C"), conditions.press)/air_density_standard_conditions

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
    return self.gross_stead_state_heating_capacity(conditions,self.cap_heating_rated[conditions.compressor_speed]) + self.fan_heat_htg(conditions) # eq. 11.31

  def heating_capacity_bins(self): #  eq. 11.117 and 11.118
    q_full_bin = {'cap': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']:
        if (t_bin >= self.temp_frost_influence_start) or (t_bin <= 17):
            q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_cond) + (self.net_total_heating_capacity(self.H1_cond) - self.net_total_heating_capacity(self.H3_cond)) * ((u(t_bin,"°F")-self.H3_cond.outdoor_drybulb)/(self.H1_cond.outdoor_drybulb-self.H3_cond.outdoor_drybulb))
        else:
            q_full_bin['cap'][q_full_bin['t_bin'].index(t_bin)] = self.net_total_heating_capacity(self.H3_cond) + (self.net_total_heating_capacity(self.H2_cond) - self.net_total_heating_capacity(self.H3_cond)) * ((u(t_bin,"°F")-self.H3_cond.outdoor_drybulb)/(self.H2_cond.outdoor_drybulb-self.H3_cond.outdoor_drybulb))
    return q_full_bin

  def heating_power_bins(self): # eq. 11.123 and 11.124
    p_full_bin = {'power': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']:
        if (t_bin >= self.temp_frost_influence_start) or (t_bin <= 17):
            p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_cond) + (self.net_heating_power(self.H1_cond) - self.net_heating_power(self.H3_cond)) * ((u(t_bin,"°F")-self.H3_cond.outdoor_drybulb)/(self.H1_cond.outdoor_drybulb-self.H3_cond.outdoor_drybulb))
        else:
            p_full_bin['power'][p_full_bin['t_bin'].index(t_bin)] = self.net_heating_power(self.H3_cond) + (self.net_heating_power(self.H2_cond) - self.net_heating_power(self.H3_cond)) * ((u(t_bin,"°F")-self.H3_cond.outdoor_drybulb)/(self.H2_cond.outdoor_drybulb-self.H3_cond.outdoor_drybulb))
    return p_full_bin

  def hp_low_temp_cutout_factor(self): # eq. 11.120 to 11.122
    delta_bin = {'delta': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    q_bin = self.heating_capacity_bins()
    p_bin = self.heating_power_bins()
    for t_bin in self.regions_table_htg['Temp']:
        if ((t_bin <= self.minimum_hp_outdoor_temp_low) or (((q_bin['cap'][q_bin['t_bin'].index(t_bin)])/(p_bin['power'][p_bin['t_bin'].index(t_bin)])) < 1)):
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 0
        elif ((t_bin <= self.minimum_hp_outdoor_temp_high) and (t_bin > self.minimum_hp_outdoor_temp_low) and (((q_bin['cap'][q_bin['t_bin'].index(t_bin)])/(p_bin['power'][p_bin['t_bin'].index(t_bin)])) >= 1)) :
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 0.5
        else:
            delta_bin['delta'][delta_bin['t_bin'].index(t_bin)] = 1
    return delta_bin

  def net_heating_power(self, conditions):
    return self.gross_stead_state_heating_power(conditions,self.cop_heating_rated[conditions.compressor_speed],self.cap_heating_rated[conditions.compressor_speed]) - self.fan_power_htg(conditions) # eq. 11.41

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
    hlf_bin = {'hlf': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    q_bin = self.heating_capacity_bins()
    bl = {'bl':self.building_load_htg(self.H1_cond),'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']:
        if q_bin['cap'][q_bin['t_bin'].index(t_bin)] > bl['bl'][bl['t_bin'].index(t_bin)]:
            hlf_bin['hlf'][hlf_bin['t_bin'].index(t_bin)] = (bl['bl'][bl['t_bin'].index(t_bin)])/(q_bin['cap'][q_bin['t_bin'].index(t_bin)])
        else:
            hlf_bin['hlf'][hlf_bin['t_bin'].index(t_bin)] = 1
    return hlf_bin

  def plf_htg(self): # eq. 11.125
    hlf = np.asarray(self.hp_htg_load_factor()['hlf'])
    plf = 1 - self.c_d_heating * (1 - hlf)
    plf = {'plf':plf,'t_bin': self.regions_table_htg['Temp']}
    return plf

  def resistance_heat(self): # eq. 11.126
    hlf = np.asarray(self.hp_htg_load_factor()['hlf'])
    bl = self.building_load_htg(self.H1_cond)
    q_bin = self.heating_capacity_bins()
    q_bin = np.asarray(q_bin['cap'])
    delta_bin = np.asarray(self.hp_low_temp_cutout_factor()['delta'])
    hrs_fraction = np.asarray(self.regions_table_htg[self.climate_region]['fraction_hrs'])
    rh = ((bl-((q_bin*hlf)*delta_bin)))*hrs_fraction
    return rh

  def bin_energy(self): # eq. 11.153 and 11.156
    e = {'e': [i*0 for i in range(18)],'t_bin': self.regions_table_htg['Temp']}
    hlf = self.hp_htg_load_factor()
    bl = {'bl':self.building_load_htg(self.H1_cond),'t_bin': self.regions_table_htg['Temp']}
    q_bin = self.heating_capacity_bins()
    p_bin = self.heating_power_bins()
    plf = self.plf_htg()
    delta_bin = self.hp_low_temp_cutout_factor()
    hrs_fraction = {'fraction': self.regions_table_htg[1]['fraction_hrs'],'t_bin': self.regions_table_htg['Temp']}
    for t_bin in self.regions_table_htg['Temp']:
        e_bin = (p_bin['power'][p_bin['t_bin'].index(t_bin)])*(hlf['hlf'][hlf['t_bin'].index(t_bin)])*(delta_bin['delta'][delta_bin['t_bin'].index(t_bin)])*(hrs_fraction['fraction'][hrs_fraction['t_bin'].index(t_bin)])
        if q_bin['cap'][q_bin['t_bin'].index(t_bin)] > bl['bl'][bl['t_bin'].index(t_bin)]:
            e_bin = e_bin/(plf['plf'][plf['t_bin'].index(t_bin)])
        e['e'][e['t_bin'].index(t_bin)] = e_bin
    return e

  def hspf(self): # eq. 11.108
    e =  np.asarray(self.bin_energy()['e'])
    hlf = np.asarray(self.hp_htg_load_factor()['hlf'])
    bl = self.building_load_htg(self.H1_cond)
    delta_bin = np.asarray(self.hp_low_temp_cutout_factor()['delta'])
    hrs_fraction = np.asarray(self.regions_table_htg[self.climate_region]['fraction_hrs'])
    rh = self.resistance_heat()
    plf = np.asarray(self.plf_htg()['plf'])
    # hspf = ((np.sum(hrs_fraction * bl))/((np.sum((((hrs_fraction * hlf) * delta_bin) * e)/plf)+(np.sum(rh))))) * self.defrost_factor()
    hspf = ((np.sum(hrs_fraction * bl))/((np.sum(e)+(np.sum(rh))))) * self.defrost_factor() # similar to 2 stage
    return convert(hspf,'','Btu/Wh')

  def writeA205(self):
    '''Write ASHRAE 205 file!!!'''
    return

def cutler_cooling_power(conditions,cop_rated,cap_rated):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([-3.437356399, 0.136656369, -0.001049231, -0.0079378, 0.000185435, -0.0001441], T_iwb, T_odb)
  eir_FF = calc_quad([1.143487507, -0.13943972, -0.004047787], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*cap_rated*(1/cop_rated)

def cutler_cooling_capacity(conditions,cap_rated):
  T_iwb = psychrolib.GetTWetBulbFromRelHum(convert(conditions.indoor_drybulb,"K","°C"),conditions.indoor_rh,conditions.press)
  T_iwb = convert(T_iwb,"°C","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([3.68637657, -0.098352478, 0.000956357, 0.005838141, -0.0000127, -0.000131702], T_iwb, T_odb)
  cap_FF = calc_quad([0.718664047, 0.41797409, -0.136638137], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*cap_rated

def cutler_heating_power(conditions,cop_rated,cap_rated):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  eir_FT = calc_biquad([0.718398423,0.003498178, 0.000142202, -0.005724331, 0.00014085, -0.000215321], T_idb, T_odb)
  eir_FF = calc_quad([2.185418751, -1.942827919, 0.757409168], conditions.mass_flow_fraction)
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return eir_FF*cap_FF*eir_FT*cap_FT*cap_rated*(1/cop_rated)

def cutler_heating_capacity(conditions,cap_rated):
  T_idb = convert(conditions.indoor_drybulb,"°K","°F") # Cutler curves use °F
  T_odb = convert(conditions.outdoor_drybulb,"K","°F") # Cutler curves use °F
  cap_FT = calc_biquad([0.566333415, -0.000744164, -0.0000103, 0.009414634, 0.0000506, -0.00000675], T_idb, T_odb)
  cap_FF = calc_quad([0.694045465, 0.474207981, -0.168253446], conditions.mass_flow_fraction)
  return cap_FF*cap_FT*cap_rated


#%%
# Move this stuff to a separate

dx_unit = DXUnit(gross_total_cooling_capacity=cutler_cooling_capacity,gross_cooling_power=cutler_cooling_power)

print(f"SEER: {dx_unit.seer()}")
print(f"Net power: {dx_unit.net_cooling_power(DXUnit.A_cond)}")
print(f"Net capacity: {dx_unit.net_total_cooling_capacity(DXUnit.A_cond)}")

# %%

dx_unit = DXUnit(gross_stead_state_heating_capacity=cutler_heating_capacity,gross_stead_state_heating_power=cutler_heating_power)

print(f"HSPF: {dx_unit.hspf()}")
print(f"Net power: {dx_unit.net_heating_power(DXUnit.H1_cond)}")
print(f"Net capacity: {dx_unit.net_total_heating_capacity(DXUnit.H1_cond)}")



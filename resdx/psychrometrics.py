from .units import fr_u, to_u

import sys
import psychrolib
psychrolib.SetUnitSystem(psychrolib.SI)

class PsychState:
  ### This class gives access to db, pressure, the additional variable given as inputs and wb. Use get function to get other variables.
  def __init__(self,drybulb,pressure=fr_u(1.0,"atm"),**kwargs):
    self.db = drybulb
    self.db_C = to_u(self.db,"°C")
    self.p = pressure
    self.wb_set = False
    self.rh_set = False
    self.hr_set = False
    self.h_set = False
    self.dp_set = False
    self.rho_set = False
    self.C_p = fr_u(1.006,"kJ/kg/K")
    if len(kwargs) > 1:
      sys.exit(f'only 1 can be provided')
    if "wetbulb" in kwargs:
      self.set_wb(kwargs["wetbulb"])
      if (self.wb_C > self.db_C) or (self.wb_C < psychrolib.GetTWetBulbFromHumRatio(self.db_C, 0, self.p)):
        sys.exit(f"Air cannot exist at these conditions: drybulb = {self.db_C}C and wetbulb = {self.wb_C}C")
    elif "hum_rat" in kwargs:
      self.set_hr(kwargs["hum_rat"])
      hr_check = psychrolib.GetHumRatioFromTWetBulb(self.db_C, self.wb_C, self.p)
      if abs(hr_check - self.hr)/hr_check > 0.01:
        sys.exit(f"Air cannot exist at these conditions: drybulb = {self.db_C}C and humidity ratio = {self.hr}")
    elif "rel_hum" in kwargs:
      self.set_rh(kwargs["rel_hum"])
      rh_check = psychrolib.GetRelHumFromTWetBulb(self.db_C, self.wb_C, self.p)
      if abs(rh_check - self.rh)/rh_check > 0.01:
        sys.exit(f"Air cannot exist at these conditions: drybulb = {self.db_C}C and relative humidity = {self.rh}")
    elif "enthalpy" in kwargs:
      self.set_h(kwargs["enthalpy"])
      h_check = psychrolib.GetMoistAirEnthalpy(self.db_C,self.get_hr())
      if abs(h_check - self.h)/h_check > 0.01:
        sys.exit(f"Air cannot exist at these conditions: drybulb = {self.db_C}C and enthalpy = {self.h}")
    elif "dew_point" in kwargs:
      self.set_dp(kwargs["dew_point"])
      dp_check = psychrolib.GetTDewPointFromTWetBulb(self.db_C,self.get_wb_C(),self.p)
      if abs(dp_check - self.dp_C)/dp_check > 0.01:
        sys.exit(f"Air cannot exist at these conditions: drybulb = {self.db_C}C and dew point = {self.dp_C}C")
    else:
      sys.exit(f'Unknonw key word argument {kwargs}.')

  def set_wb(self, wb):
    self.wb = wb
    self.wb_C = to_u(self.wb,"°C")
    self.wb_set = True
    return self.wb

  def set_dp(self, dp):
    self.dp = dp
    self.dp_C = to_u(self.dp,"°C")
    if not self.wb_set:
      self.set_wb(fr_u(psychrolib.GetTWetBulbFromTDewPoint(self.db_C, self.dp_C, self.p) ,"°C"))
    self.dp_set = True
    return self.dp

  def set_rh(self, rh):
    self.rh = rh
    if not self.wb_set:
      self.set_wb(fr_u(psychrolib.GetTWetBulbFromRelHum(self.db_C, self.rh, self.p),"°C"))
    self.rh_set = True
    return self.rh

  def set_hr(self, hr):
    self.hr = hr
    if not self.wb_set:
      self.set_wb(fr_u(psychrolib.GetTWetBulbFromHumRatio(self.db_C, self.hr, self.p),"°C"))
    self.hr_set = True
    return self.hr

  def set_h(self, h):
    self.h = h
    if not self.hr_set:
      self.set_hr(psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(self.h, self.db_C))
    self.h_set = True
    return self.h

  def get_wb(self):
    if self.wb_set:
      return self.wb
    else:
      sys.exit(f'Wetbulb not set')

  def get_wb_C(self):
    if self.wb_set:
      return self.wb_C
    else:
      sys.exit(f'Wetbulb not set')

  def get_dp(self):
    if not self.dp_set:
      self.set_dp(psychrolib.GetTDewPointFromTWetBulb(self.db_C,self.get_wb_C(),self.p))
    return self.dp

  def get_dp_C(self):
    if not self.dp_set:
      self.set_dp(to_u(psychrolib.GetTDewPointFromTWetBulb(self.db_C,self.get_wb_C(),self.p),"°C"))
    return self.dp_C

  def set_rho(self, rho):
    self.rho = rho
    self.rho_set = True
    return self.rho

  def get_rho(self):
    if self.rho_set:
      return self.rho
    else:
      return self.set_rho(psychrolib.GetMoistAirDensity(self.db_C, self.get_hr(), self.p))

  def get_hr(self):
    if self.hr_set:
      return self.hr
    else:
      return self.set_hr(psychrolib.GetHumRatioFromTWetBulb(self.db_C, self.get_wb_C(), self.p))

  def get_rh(self):
    if self.rh_set:
        return self.rh
    else:
      return self.set_rh(psychrolib.GetRelHumFromTWetBulb(self.db_C,self.get_wb_C(),self.p))

  def get_h(self):
    if self.h_set:
        return self.h
    else:
      return self.set_h(psychrolib.GetMoistAirEnthalpy(self.db_C,self.get_hr()))

STANDARD_CONDITIONS = PsychState(drybulb=fr_u(70.0,"°F"),hum_rat=1e-07) # The min HR in psychrolib is 1e-07

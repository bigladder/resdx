from koozie import fr_u, to_u

import sys
import psychrolib
psychrolib.SetUnitSystem(psychrolib.SI)

class PsychState:
  def __init__(self,drybulb,pressure=fr_u(1.0,"atm"),**kwargs):
    self.db = drybulb
    self.db_C = to_u(self.db,"°C")
    self.p = pressure
    self.wb_set = False
    self.rh_set = False
    self.hr_set = False
    self.dp_set = False
    self.h_set = False
    self.rho_set = False
    self.C_p = fr_u(1.006,"kJ/kg/K")
    if len(kwargs) > 1:
      sys.exit(f'only 1 can be provided')
    if "wetbulb" in kwargs:
      self.set_wb(kwargs["wetbulb"])
    elif "hum_rat" in kwargs:
      self.set_hr(kwargs["hum_rat"])
    elif "rel_hum" in kwargs:
      self.set_rh(kwargs["rel_hum"])
    elif "enthalpy" in kwargs:
      self.set_h(kwargs["enthalpy"])
    else:
      sys.exit(f'Unknown or missing key word argument {kwargs}.')

  def set_wb(self, wb):
    self.wb = wb
    self.wb_C = to_u(self.wb,"°C")
    self.wb_set = True
    return self.wb

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

  def get_h(self):
    if self.h_set:
        return self.h
    else:
      return self.set_h(psychrolib.GetMoistAirEnthalpy(self.db_C,self.get_hr()))


STANDARD_CONDITIONS = PsychState(drybulb=fr_u(70.0,"°F"),hum_rat=0.0)


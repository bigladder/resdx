from .psychrometrics import PsychState, STANDARD_CONDITIONS, psychrolib
from .units import fr_u

class OperatingConditions:
  def __init__(self, outdoor=STANDARD_CONDITIONS,
                     indoor=STANDARD_CONDITIONS,
                     compressor_speed=0):
    self.outdoor = outdoor
    self.indoor = indoor
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)
    self.rated_air_flow_set = False

  def set_rated_air_flow(self, air_vol_flow_rated_per_rated_cap, net_total_cooling_capacity_rated, air_vol_flow_per_rated_cap = None, air_vol_flow = None):
    self.net_total_cooling_capacity_rated = net_total_cooling_capacity_rated
    self.std_air_vol_flow_rated = air_vol_flow_rated_per_rated_cap*net_total_cooling_capacity_rated
    self.air_vol_flow_rated = self.std_air_vol_flow_rated*STANDARD_CONDITIONS.get_rho()/self.indoor.get_rho() # Shouldn't we calculate at normal conditions first then get std?
    self.air_mass_flow_rated = self.air_vol_flow_rated*self.indoor.get_rho()
    self.rated_air_flow_set = True
    if air_vol_flow is None:
      if air_vol_flow_per_rated_cap is not None:
        self.set_air_flow(air_vol_flow_per_rated_cap*net_total_cooling_capacity_rated)
      else:
        self.set_air_flow(self.air_vol_flow_rated)
    else:
      self.set_air_flow(air_vol_flow)

  def set_air_flow(self, air_vol_flow):
    self.air_vol_flow = air_vol_flow
    self.air_mass_flow = air_vol_flow*self.indoor.get_rho()
    if self.rated_air_flow_set:
      self.std_air_vol_flow = self.air_mass_flow/STANDARD_CONDITIONS.get_rho()
      self.air_mass_flow_fraction = self.air_mass_flow/self.air_mass_flow_rated
      self.std_air_vol_flow_per_capacity = self.std_air_vol_flow/self.net_total_cooling_capacity_rated # Is this std air flow/ capacity or per capacity rated?

class CoolingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=fr_u(95.0,"°F"),wetbulb=fr_u(75.0,"°F")),
                    indoor=PsychState(drybulb=fr_u(80.0,"°F"),wetbulb=fr_u(67.0,"°F")),
                    compressor_speed=0):
    super().__init__(outdoor, indoor, compressor_speed)

class HeatingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=fr_u(47.0,"°F"),wetbulb=fr_u(43.0,"°F")),
                    indoor=PsychState(drybulb=fr_u(70.0,"°F"),wetbulb=fr_u(60.0,"°F")),
                    compressor_speed=0):
    super().__init__(outdoor, indoor, compressor_speed)

from .psychrometrics import PsychState, STANDARD_CONDITIONS, psychrolib
from .units import fr_u

class OperatingConditions:
  def __init__(self, outdoor=STANDARD_CONDITIONS,
                     indoor=STANDARD_CONDITIONS,
                     compressor_speed=0,
                     rated_flow_external_static_pressure=None):
    self.outdoor = outdoor
    self.indoor = indoor
    self.compressor_speed = compressor_speed # compressor speed index (0 = full speed, 1 = next lowest, ...)
    self.rated_air_flow_set = False
    self.rated_flow_external_static_pressure = rated_flow_external_static_pressure

  def set_rated_air_flow(self, air_vol_flow_rated, net_total_cooling_capacity_rated):
    self.net_total_cooling_capacity_rated = net_total_cooling_capacity_rated
    self.std_air_vol_flow_rated = air_vol_flow_rated
    self.air_vol_flow_rated = self.std_air_vol_flow_rated*STANDARD_CONDITIONS.get_rho()/self.indoor.get_rho()
    self.air_mass_flow_rated = self.air_vol_flow_rated*self.indoor.get_rho()
    self.rated_air_flow_set = True
    self.set_air_flow(self.air_vol_flow_rated)

  def set_air_flow(self, air_vol_flow):
    self.air_vol_flow = air_vol_flow
    self.air_mass_flow = air_vol_flow*self.indoor.get_rho()
    if self.rated_air_flow_set:
      self.std_air_vol_flow = self.air_mass_flow/STANDARD_CONDITIONS.get_rho()
      self.air_mass_flow_ratio = self.air_mass_flow/self.air_mass_flow_rated
      self.std_air_vol_flow_per_capacity = self.std_air_vol_flow/self.net_total_cooling_capacity_rated
      if self.rated_flow_external_static_pressure is not None:
        self.external_static_pressure = self.rated_flow_external_static_pressure*(self.air_mass_flow_ratio)**2.

  def set_air_mass_flow_ratio(self, air_mass_flow_ratio):
    self.air_mass_flow_ratio = air_mass_flow_ratio
    if self.rated_flow_external_static_pressure is not None:
      self.external_static_pressure = self.rated_flow_external_static_pressure*(self.air_mass_flow_ratio)**2.
    if self.rated_air_flow_set:
      self.air_mass_flow = self.air_mass_flow_ratio*self.air_mass_flow_rated
      self.air_vol_flow = self.air_mass_flow/self.indoor.get_rho()
      self.std_air_vol_flow = self.air_mass_flow/STANDARD_CONDITIONS.get_rho()
      self.std_air_vol_flow_per_capacity = self.std_air_vol_flow/self.net_total_cooling_capacity_rated

class CoolingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=fr_u(95.0,"°F"),wetbulb=fr_u(75.0,"°F")),
                    indoor=PsychState(drybulb=fr_u(80.0,"°F"),wetbulb=fr_u(67.0,"°F")),
                    compressor_speed=0,
                    rated_flow_external_static_pressure=fr_u(0.2, "in_H2O")):
    super().__init__(outdoor, indoor, compressor_speed, rated_flow_external_static_pressure)

class HeatingConditions(OperatingConditions):
  def __init__(self,outdoor=PsychState(drybulb=fr_u(47.0,"°F"),wetbulb=fr_u(43.0,"°F")),
                    indoor=PsychState(drybulb=fr_u(70.0,"°F"),wetbulb=fr_u(60.0,"°F")),
                    compressor_speed=0,
                    rated_flow_external_static_pressure=fr_u(0.2, "in_H2O")):
    super().__init__(outdoor, indoor, compressor_speed, rated_flow_external_static_pressure)

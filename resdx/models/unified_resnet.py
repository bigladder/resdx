from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .henderson_defrost_model import HendersonDefrostModel
from ..units import fr_u, to_u

def fan_efficacy(seer):
    if seer <= 14:
        return fr_u(0.25,'W/(cu_ft/min)')
    elif seer >= 16:
        return fr_u(0.18,'W/(cu_ft/min)')
    else:
        return fr_u(0.25,'W/(cu_ft/min)') + (fr_u(0.18,'W/(cu_ft/min)') - fr_u(0.25,'W/(cu_ft/min)'))/2.0 * (seer - 14.0)


def c_d(seer):
    if seer <= 12:
        return 0.2
    elif seer >= 13:
        return 0.1
    else:
        return 0.2 + (0.1 - 0.2)*(seer - 12.0)

def estimated_seer(hspf): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
    return (hspf - 3.2627)/0.3526


class RESNETDXModel(DXModel):

  # Power and capacity
  def gross_cooling_power(self, conditions):
    return NRELDXModel.gross_cooling_power(self, conditions)

  def gross_total_cooling_capacity(self, conditions):
    return NRELDXModel.gross_total_cooling_capacity(self, conditions)

  def gross_sensible_cooling_capacity(self, conditions):
    return NRELDXModel.gross_sensible_cooling_capacity(self, conditions)

  def gross_shr(self, conditions):
    return Title24DXModel.gross_shr(self, conditions)

  def gross_steady_state_heating_capacity(self, conditions):
    return NRELDXModel.gross_steady_state_heating_capacity(self, conditions)

  def gross_integrated_heating_capacity(self, conditions):
    return HendersonDefrostModel.gross_integrated_heating_capacity(self, conditions)

  def gross_steady_state_heating_power(self, conditions):
    return NRELDXModel.gross_steady_state_heating_power(self, conditions)

  def gross_integrated_heating_power(self, conditions):
    return HendersonDefrostModel.gross_integrated_heating_power(self, conditions)

  # Default assumptions
  def set_flow_rated_per_cap_cooling_rated(self, input):
    default = fr_u(375.0,"(cu_ft/min)/ton_of_refrigeration")
    if self.system.number_of_input_stages == 1:
      self.system.flow_rated_per_cap_cooling_rated = self.set_default(input, [default])
    elif self.system.number_of_input_stages == 2:
      self.system.flow_rated_per_cap_cooling_rated = self.set_default(input, [default, default*0.86])
    else:
      self.system.flow_rated_per_cap_cooling_rated = self.set_default(input, [default]*self.system.number_of_input_stages)

  def set_flow_rated_per_cap_heating_rated(self, input):
    default = fr_u(375.0,"(cu_ft/min)/ton_of_refrigeration")
    if self.system.number_of_input_stages == 1:
      self.system.flow_rated_per_cap_heating_rated = self.set_default(input, [default])
    elif self.system.number_of_input_stages == 2:
      self.system.flow_rated_per_cap_heating_rated = self.set_default(input, [default, default*0.8])
    else:
      self.system.flow_rated_per_cap_heating_rated = self.set_default(input, [default]*self.system.number_of_input_stages)

  def set_net_total_cooling_capacity_rated(self, input):
    NRELDXModel.set_net_total_cooling_capacity_rated(self, input)

  def set_net_heating_capacity_rated(self, input):
    input = self.set_default(input, self.system.net_total_cooling_capacity_rated[0]*0.98 + fr_u(180.,"Btu/hr")) # From Title24
    if type(input) is list:
      self.system.net_heating_capacity_rated = input
    else:
      # From NREL
      if self.system.number_of_input_stages == 1:
        self.system.net_heating_capacity_rated = [input]
      elif self.system.number_of_input_stages == 2:
        fan_power_0 = input*self.system.fan_efficacy_heating_rated[0]*self.system.flow_rated_per_cap_heating_rated[0]
        gross_cap_0 = input - fan_power_0
        gross_cap_1 = gross_cap_0*0.72
        net_cap_1 = gross_cap_1/(1. - self.system.fan_efficacy_heating_rated[1]*self.system.flow_rated_per_cap_heating_rated[1])
        self.system.net_heating_capacity_rated = [input, net_cap_1]

  def set_c_d_cooling(self, input):
    self.system.c_d_cooling = self.set_default(input, 0.1)

  def set_c_d_heating(self, input):
    self.system.c_d_heating = self.set_default(input, 0.142)

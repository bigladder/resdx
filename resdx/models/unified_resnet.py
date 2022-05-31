from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .henderson_defrost_model import HendersonDefrostModel
from ..units import fr_u
from ..fan import ConstantEfficacyFan

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

  def gross_cooling_power_charge_factor(self, conditions):
    return NRELDXModel.gross_cooling_power_charge_factor(self, conditions)

  def gross_total_cooling_capacity_charge_factor(self, conditions):
    return NRELDXModel.gross_total_cooling_capacity_charge_factor(self, conditions)

  def gross_steady_state_heating_capacity_charge_factor(self, conditions):
    return NRELDXModel.gross_steady_state_heating_capacity_charge_factor(self, conditions)

  def gross_steady_state_heating_power_charge_factor(self, conditions):
    return NRELDXModel.gross_steady_state_heating_power_charge_factor(self, conditions)

  # Default assumptions
  def set_rated_fan_characteristics(self, fan):
    if fan is not None:
      pass
    else:
      if self.system.input_seer is not None:
        cooling_fan_efficacy = RESNETDXModel.fan_efficacy(self.system.input_seer)
      else:
        cooling_fan_efficacy = fr_u(0.25,'W/cfm')
      if self.system.input_hspf is not None:
        heating_fan_efficacy = RESNETDXModel.fan_efficacy(RESNETDXModel.estimated_seer(self.system.input_hspf))
      else:
        heating_fan_efficacy = fr_u(0.25,'W/cfm')
      self.system.fan_efficacy_cooling_rated = [cooling_fan_efficacy]*self.system.number_of_input_stages
      self.system.fan_efficacy_heating_rated = [heating_fan_efficacy]*self.system.number_of_input_stages

      # Airflows
      cooling_default = fr_u(375.,"cfm/ton_ref")
      heating_default = fr_u(375.,"cfm/ton_ref")

      if self.system.number_of_input_stages == 1:
        self.system.flow_rated_per_cap_cooling_rated = [cooling_default]
        self.system.flow_rated_per_cap_heating_rated = [heating_default]
      elif self.system.number_of_input_stages == 2:
        self.system.flow_rated_per_cap_cooling_rated = [cooling_default, cooling_default*0.86]
        self.system.flow_rated_per_cap_heating_rated = [heating_default, heating_default*0.8]

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

  def set_fan(self, input):
    if input is not None:
      # TODO: Handle default mappings?
      self.system.fan = input
    else:
      airflows = []
      efficacies = []
      fan_speed = 0
      if self.system.cooling_fan_speed_mapping is None:
        set_cooling_fan_speed_map = True
        self.system.cooling_fan_speed_mapping = []

      if self.system.heating_fan_speed_mapping is None:
        set_heating_fan_speed_map = True
        self.system.heating_fan_speed_mapping = []

      for i, cap in enumerate(self.system.net_total_cooling_capacity_rated):
        airflows.append(cap*self.system.flow_rated_per_cap_cooling_rated[i])
        efficacies.append(self.system.fan_efficacy_cooling_rated[i])
        if set_cooling_fan_speed_map:
          self.system.cooling_fan_speed_mapping.append(fan_speed)
          fan_speed += 1

      for i, cap in enumerate(self.system.net_total_cooling_capacity_rated):
        airflows.append(cap*self.system.flow_rated_per_cap_heating_rated[i])
        efficacies.append(self.system.fan_efficacy_heating_rated[i])
        if set_heating_fan_speed_map:
          self.system.heating_fan_speed_mapping.append(fan_speed)
          fan_speed += 1

      fan = ConstantEfficacyFan(airflows, fr_u(0.20, "in_H2O"), design_efficacy=efficacies)
      self.system.fan = fan

  def set_net_capacities_and_fan(self, net_total_cooling_capacity_rated, net_heating_capacity_rated, fan):
    self.set_rated_fan_characteristics(fan)
    self.set_net_total_cooling_capacity_rated(net_total_cooling_capacity_rated)
    self.set_net_heating_capacity_rated(net_heating_capacity_rated)
    self.set_fan(fan)

  def set_c_d_cooling(self, input):
    if self.system.input_seer is None:
      default = 0.25
    else:
      default = RESNETDXModel.c_d(self.system.input_seer)
    self.system.c_d_cooling = self.set_default(input, default)

  def set_c_d_heating(self, input):
    if self.system.input_hspf is None:
      default = 0.25
    else:
      default = RESNETDXModel.c_d(RESNETDXModel.estimated_seer(self.system.input_hspf))
    self.system.c_d_heating = self.set_default(input, default)

  def set_net_cooling_cop_rated(self, input):
    NRELDXModel.set_net_cooling_cop_rated(self, input)

  def set_gross_cooling_cop_rated(self, input):
    NRELDXModel.set_gross_cooling_cop_rated(self, input)

  def set_net_heating_cop_rated(self, input):
    NRELDXModel.set_net_heating_cop_rated(self, input)

  def set_gross_heating_cop_rated(self, input):
    NRELDXModel.set_gross_heating_cop_rated(self, input)

  @staticmethod
  def fan_efficacy(seer):
      if seer <= 14:
          return fr_u(0.25,'W/cfm')
      elif seer >= 16:
          return fr_u(0.18,'W/cfm')
      else:
          return fr_u(0.25,'W/cfm') + (fr_u(0.18,'W/cfm') - fr_u(0.25,'W/cfm'))/2.0 * (seer - 14.0)

  @staticmethod
  def c_d(seer):
      if seer <= 12:
          return 0.2
      elif seer >= 13:
          return 0.1
      else:
          return 0.2 + (0.1 - 0.2)*(seer - 12.0)

  @staticmethod
  def estimated_seer(hspf): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
      return (hspf - 3.2627)/0.3526

  @staticmethod
  def estimated_hspf(seer): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
      return seer*0.3526 + 3.2627


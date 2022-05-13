from ..units import fr_u, to_u


class DXModel:

  def __init__(self):
    self.system = None

  def set_system(self, system):
    self.system = system
    if "base_model" in system.kwargs:
      system.kwargs["base_model"].system = system

  # Power and capacity
  def gross_cooling_power(self, conditions):
    raise NotImplementedError()

  def gross_total_cooling_capacity(self, conditions):
    raise NotImplementedError()

  def gross_sensible_cooling_capacity(self, conditions):
    raise NotImplementedError()

  def gross_shr(self, conditions):
    '''This is used to calculate the SHR at rated conditions'''
    raise NotImplementedError()

  def gross_steady_state_heating_capacity(self, conditions):
    raise NotImplementedError()

  def gross_integrated_heating_capacity(self, conditions):
    raise NotImplementedError()

  def gross_steady_state_heating_power(self, conditions):
    raise NotImplementedError()

  def gross_integrated_heating_power(self, conditions):
    raise NotImplementedError()

  # Default assumptions
  def set_default(self, input, default):
    if input is None:
      return default
    else:
      if type(default) is list:
        if type(input) is list:
          return input
        else:
          return [input]*self.system.number_of_input_stages
      else:
        return input

  def set_fan_efficacy_cooling_rated(self, input):
    self.system.fan_efficacy_cooling_rated = self.set_default(input, [fr_u(0.25,'W/(cu_ft/min)')]*self.system.number_of_input_stages)

  def set_fan_efficacy_heating_rated(self, input):
    self.system.fan_efficacy_heating_rated = self.set_default(input, [fr_u(0.25,'W/(cu_ft/min)')]*self.system.number_of_input_stages)

  def set_flow_rated_per_cap_cooling_rated(self, input):
    self.system.flow_rated_per_cap_cooling_rated = self.set_default(input, [fr_u(375.0,"(cu_ft/min)/ton_of_refrigeration")]*self.system.number_of_input_stages)

  def set_flow_rated_per_cap_heating_rated(self, input):
    self.system.flow_rated_per_cap_heating_rated = self.set_default(input, [fr_u(375.0,"(cu_ft/min)/ton_of_refrigeration")]*self.system.number_of_input_stages) # TODO: Check assumption

  def set_net_total_cooling_capacity_rated(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.net_total_cooling_capacity_rated = input
    else:
      self.system.net_total_cooling_capacity_rated = [input]*self.system.number_of_input_stages

  def set_net_heating_capacity_rated(self, input):
    input = self.set_default(input, self.system.net_total_cooling_capacity_rated[0])
    if type(input) is list:
      self.system.net_heating_capacity_rated = input
    else:
      self.system.net_heating_capacity_rated = [input]*self.system.number_of_input_stages

  def set_c_d_cooling(self, input):
    self.system.c_d_cooling = self.set_default(input, 0.25)

  def set_c_d_heating(self, input):
    self.system.c_d_heating = self.set_default(input, 0.25)


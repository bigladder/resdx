from ..units import fr_u
from ..fan import ConstantEfficacyFan

class DXModel:

  def __init__(self):
    self.system = None
    self.allowed_kwargs = ["base_model"]

  def set_system(self, system):
    self.system = system
    for kwarg in system.kwargs:
      if kwarg not in self.allowed_kwargs:
        raise Exception(f"Unrecognized key word argument: {kwarg}")
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

  def gross_total_cooling_capacity_charge_factor(self, conditions):
    return 1.0

  def gross_total_cooling_capacity_charge_factor(self, conditions):
    return 1.0

  def gross_cooling_power_charge_factor(self, conditions):
    return 1.0

  def gross_steady_state_heating_capacity_charge_factor(self, conditions):
    return 1.0

  def gross_steady_state_heating_power_charge_factor(self, conditions):
    return 1.0

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

  def set_fan(self, input):
    if input is not None:
      # TODO: Handle default mappings?
      self.system.fan = input
    else:
      air_flows = []
      for cap in self.system.net_total_cooling_capacity_rated:
        air_flows.append(cap*fr_u(375.0,"cfm/ton_ref"))
      fan = ConstantEfficacyFan(air_flows, fr_u(0.20, "in_H2O"), efficacy_design=fr_u(0.25,'W/cfm'))
      self.system.fan = fan

      if self.system.heating_fan_speed_mapping is None:
        self.system.heating_fan_speed_mapping = list(range(fan.number_of_speeds))

      if self.system.cooling_fan_speed_mapping is None:
        self.system.cooling_fan_speed_mapping = list(range(fan.number_of_speeds))

  def set_net_capacities_and_fan(self, net_total_cooling_capacity_rated, net_heating_capacity_rated, fan):
    self.set_net_total_cooling_capacity_rated(net_total_cooling_capacity_rated)
    self.set_net_heating_capacity_rated(net_heating_capacity_rated)
    self.set_fan(fan)

  def set_c_d_cooling(self, input):
    self.system.c_d_cooling = self.set_default(input, 0.25)

  def set_c_d_heating(self, input):
    self.system.c_d_heating = self.set_default(input, 0.25)

  def set_net_cooling_cop_rated(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.net_cooling_cop_rated = input
    else:
      self.system.net_cooling_cop_rated = [input]*self.system.number_of_input_stages
    self.system.net_cooling_power_rated = [self.system.net_total_cooling_capacity_rated[i]/self.system.net_cooling_cop_rated[i] for i in range(self.system.number_of_input_stages)]
    self.system.gross_cooling_power_rated = [self.system.net_cooling_power_rated[i] - self.system.cooling_fan_power_rated[i] for i in range(self.system.number_of_input_stages)]
    self.system.gross_cooling_cop_rated = [self.system.gross_total_cooling_capacity_rated[i]/self.system.gross_cooling_power_rated[i] for i in range(self.system.number_of_input_stages)]

  def set_gross_cooling_cop_rated(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.gross_cooling_cop_rated = input
    else:
      self.system.gross_cooling_cop_rated = [input]*self.system.number_of_input_stages
      self.system.gross_cooling_power_rated = [self.system.gross_total_cooling_capacity_rated[i]/self.system.gross_cooling_cop_rated[i] for i in range(self.system.number_of_input_stages)]
      self.system.net_cooling_power_rated = [self.system.gross_cooling_power_rated[i] + self.system.cooling_fan_power_rated[i] for i in range(self.system.number_of_input_stages)]
      self.system.net_cooling_cop_rated = [self.system.net_total_cooling_capacity_rated[i]/self.system.net_cooling_power_rated[i] for i in range(self.system.number_of_input_stages)]

  def set_net_heating_cop_rated(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.net_heating_cop_rated = input
    else:
      self.system.net_heating_cop_rated = [input]*self.system.number_of_input_stages
    self.system.net_heating_power_rated = [self.system.net_heating_capacity_rated[i]/self.system.net_heating_cop_rated[i] for i in range(self.system.number_of_input_stages)]
    self.system.gross_heating_power_rated = [self.system.net_heating_power_rated[i] - self.system.heating_fan_power_rated[i] for i in range(self.system.number_of_input_stages)]
    self.system.gross_heating_cop_rated = [self.system.gross_heating_capacity_rated[i]/self.system.gross_heating_power_rated[i] for i in range(self.system.number_of_input_stages)]

  def set_gross_heating_cop_rated(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.gross_heating_cop_rated = input
    else:
      self.system.gross_heating_cop_rated = [input]*self.system.number_of_input_stages
      self.system.gross_heating_power_rated = [self.system.gross_heating_capacity_rated[i]/self.system.gross_heating_cop_rated[i] for i in range(self.system.number_of_input_stages)]
      self.system.net_heating_power_rated = [self.system.gross_heating_power_rated[i] + self.system.heating_fan_power_rated[i] for i in range(self.system.number_of_input_stages)]
      self.system.net_heating_cop_rated = [self.system.net_heating_capacity_rated[i]/self.system.net_heating_power_rated[i] for i in range(self.system.number_of_input_stages)]

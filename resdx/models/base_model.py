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

  def set_rated_net_total_cooling_capacity(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_total_cooling_capacity = input
    else:
      self.system.rated_net_total_cooling_capacity = [input]*self.system.number_of_input_stages

  def set_rated_net_heating_capacity(self, input):
    input = self.set_default(input, self.system.rated_net_total_cooling_capacity[0])
    if type(input) is list:
      self.system.rated_net_heating_capacity = input
    else:
      self.system.rated_net_heating_capacity = [input]*self.system.number_of_input_stages

  def set_fan(self, input):
    if input is not None:
      # TODO: Handle default mappings?
      self.system.fan = input
    else:
      airflows = []
      self.system.cooling_fan_speed = []
      self.system.heating_fan_speed = []
      for net_capacity in self.system.rated_net_total_cooling_capacity:
        airflows.append(net_capacity*fr_u(375.0,"cfm/ton_ref"))
        self.system.cooling_fan_speed.append(len(airflows) - 1)

      for net_capacity in self.system.rated_net_heating_capacity:
        airflows.append(net_capacity*fr_u(375.0,"cfm/ton_ref"))
        self.system.heating_fan_speed.append(len(airflows) - 1)

      fan = ConstantEfficacyFan(airflows, fr_u(0.50, "in_H2O"), design_efficacy=fr_u(0.365,'W/cfm'))
      self.system.fan = fan

  def set_net_capacities_and_fan(self, rated_net_total_cooling_capacity, rated_net_heating_capacity, fan):
    self.set_rated_net_total_cooling_capacity(rated_net_total_cooling_capacity)
    self.set_rated_net_heating_capacity(rated_net_heating_capacity)
    self.set_fan(fan)

  def set_c_d_cooling(self, input):
    self.system.c_d_cooling = self.set_default(input, 0.25)

  def set_c_d_heating(self, input):
    self.system.c_d_heating = self.set_default(input, 0.25)

  def set_rated_net_cooling_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_cooling_cop = input
    else:
      self.system.rated_net_cooling_cop = [input]*self.system.number_of_input_stages
    self.system.rated_net_cooling_power = [self.system.rated_net_total_cooling_capacity[i]/self.system.rated_net_cooling_cop[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_gross_cooling_power = [self.system.rated_net_cooling_power[i] - self.system.rated_cooling_fan_power[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_gross_cooling_cop = [self.system.rated_gross_total_cooling_capacity[i]/self.system.rated_gross_cooling_power[i] for i in range(self.system.number_of_input_stages)]

  def set_rated_gross_cooling_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_gross_cooling_cop = input
    else:
      self.system.rated_gross_cooling_cop = [input]*self.system.number_of_input_stages
      self.system.rated_gross_cooling_power = [self.system.rated_gross_total_cooling_capacity[i]/self.system.rated_gross_cooling_cop[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_net_cooling_power = [self.system.rated_gross_cooling_power[i] + self.system.rated_cooling_fan_power[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_net_cooling_cop = [self.system.rated_net_total_cooling_capacity[i]/self.system.rated_net_cooling_power[i] for i in range(self.system.number_of_input_stages)]

  def set_rated_net_heating_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_net_heating_cop = input
    else:
      self.system.rated_net_heating_cop = [input]*self.system.number_of_input_stages
    self.system.rated_net_heating_power = [self.system.rated_net_heating_capacity[i]/self.system.rated_net_heating_cop[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_gross_heating_power = [self.system.rated_net_heating_power[i] - self.system.rated_heating_fan_power[i] for i in range(self.system.number_of_input_stages)]
    self.system.rated_gross_heating_cop = [self.system.rated_gross_heating_capacity[i]/self.system.rated_gross_heating_power[i] for i in range(self.system.number_of_input_stages)]

  def set_rated_gross_heating_cop(self, input):
    # No default, but need to set to list (and default lower speeds)
    if type(input) is list:
      self.system.rated_gross_heating_cop = input
    else:
      self.system.rated_gross_heating_cop = [input]*self.system.number_of_input_stages
      self.system.rated_gross_heating_power = [self.system.rated_gross_heating_capacity[i]/self.system.rated_gross_heating_cop[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_net_heating_power = [self.system.rated_gross_heating_power[i] + self.system.rated_heating_fan_power[i] for i in range(self.system.number_of_input_stages)]
      self.system.rated_net_heating_cop = [self.system.rated_net_heating_capacity[i]/self.system.rated_net_heating_power[i] for i in range(self.system.number_of_input_stages)]

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

  # Defaults
  @staticmethod
  def set_default(input, default):
    if input is None:
      return default
    else:
      return input

  def set_c_d_cooling(self, input):
    self.system.c_d_cooling = DXModel.set_default(input, 0.25)



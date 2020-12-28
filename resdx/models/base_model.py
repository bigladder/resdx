class DXModel:

  @staticmethod
  def gross_cooling_power(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_total_cooling_capacity(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_sensible_cooling_capacity(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_shr(conditions):
    '''This is used to calculate the SHR at rated conditions'''
    raise NotImplementedError()

  @staticmethod
  def gross_steady_state_heating_capacity(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    raise NotImplementedError()

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    raise NotImplementedError()
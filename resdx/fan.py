from .units import fr_u
from math import exp, log, inf
from scipy import optimize # Used for finding system/fan curve intersection

class Fan:
  def __init__(
    self,
    design_airflow,
    design_external_static_pressure,
    design_efficacy=fr_u(0.365,'W/cfm')): # AHRI 210/240 2017 default
      if type(design_airflow) is list:
        self.design_airflow = design_airflow
      else:
        self.design_airflow = [design_airflow]
      self.number_of_speeds = len(self.design_airflow)
      self.design_external_static_pressure = design_external_static_pressure
      if type(design_efficacy) is list:
        self.design_efficacy = design_efficacy
      else:
        self.design_efficacy = [design_efficacy]*self.number_of_speeds

      self.system_exponent = 0.5
      self.system_curve_constant = self.design_external_static_pressure**(self.system_exponent)/self.design_airflow[0]

    # system curve function: given flow rate, calculate pressure
  def system_pressure(self, airflow):
    return (airflow*self.system_curve_constant)**(1./self.system_exponent)

  def system_flow(self, external_static_pressure):
    return external_static_pressure**(self.system_exponent)/self.system_curve_constant

  def efficacy(self, speed_setting, external_static_pressure=None):
    raise NotImplementedError()

  def airflow(self, speed_setting, external_static_pressure=None):
    raise NotImplementedError()

  def power(self, speed_setting, external_static_pressure=None):
    return self.airflow(speed_setting, external_static_pressure)*self.efficacy(speed_setting, external_static_pressure)

  def rotational_speed(self, speed_setting, external_static_pressure=None):
    raise NotImplementedError()

  def operating_pressure(self, speed_setting, system_curve=None):
    # Calculate pressure that corresponds to intersection of system curve and fan curve for this setting
    if system_curve is None:
      fx = self.system_flow
    else:
      fx = system_curve
    p, solution = optimize.brentq(lambda x : self.airflow(speed_setting, x) - fx(x), 0., 2.*self.design_external_static_pressure, full_output = True)
    return p

  def write_A205(self):
    pass

class ConstantEfficacyFan(Fan):
  def efficacy(self, speed_setting, external_static_pressure=None):
    return self.design_efficacy[speed_setting]

  def airflow(self, speed_setting, external_static_pressure=None):
    return self.design_airflow[speed_setting]

class PSCFan(Fan):
  '''Based largely on measured fan performance by Proctor Engineering'''
  def __init__(
    self,
    design_airflow,
    design_external_static_pressure=fr_u(0.5, "in_H2O"),
    design_efficacy=fr_u(0.365,'W/cfm')):
      super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
      self.AIRFLOW_COEFFICIENT = fr_u(10.,'cfm')
      self.AIRFLOW_EXP_COEFFICIENT = fr_u(5.355391179,'1/in_H2O')
      self.EFFICACY_SLOPE = -0.1446674009  # Relative change in efficacy at lower flow ratios
      design_airflow_reduction = self.airflow_reduction(design_external_static_pressure)
      self.airflow_free = [self.design_airflow[i] + design_airflow_reduction for i in range(self.number_of_speeds)]
      self.airflow_ratios = [self.airflow_free[i]/self.airflow_free[0] for i in range(self.number_of_speeds)]
      self.design_efficacy = design_efficacy
      self.efficacies = [design_efficacy*(1. + self.EFFICACY_SLOPE*(self.airflow_ratios[i] - 1.)) for i in range(self.number_of_speeds)]
      self.block_pressure = [log(self.airflow_free[i]/self.AIRFLOW_COEFFICIENT + 1.)/self.AIRFLOW_EXP_COEFFICIENT for i in range(self.number_of_speeds)]
      self.speed_free = [fr_u(1040.,'rpm')*self.airflow_ratios[i] for i in range(self.number_of_speeds)]

  def efficacy(self, speed_setting, external_static_pressure=None):
      return self.efficacies[speed_setting]

  def airflow(self, speed_setting, external_static_pressure=None):
    i = speed_setting
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return max(self.airflow_free[i] - self.airflow_reduction(external_static_pressure),0.)

  def rotational_speed(self, speed_setting, external_static_pressure=None):
    i = speed_setting
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    if external_static_pressure > self.block_pressure[i]:
      return fr_u(1100.,'rpm')
    else:
      return self.speed_free[i] + (fr_u(1100.,'rpm') - self.speed_free[i])*external_static_pressure/self.block_pressure[i]

  def airflow_reduction(self, external_static_pressure):
    return self.AIRFLOW_COEFFICIENT*(exp(external_static_pressure*self.AIRFLOW_EXP_COEFFICIENT)-1.)

  def operating_pressure(self, speed_setting, system_curve=None):
    if system_curve is None:
      # TODO Solve algebraically for improved performance
      return super().operating_pressure(speed_setting, system_curve)
    else:
      return super().operating_pressure(speed_setting, system_curve)

    # Solve algebraically
    pass

class ECMFlowFan(Fan):
  '''Constant flow ECM fan. Based largely on measured fan performance by Proctor Engineering'''
  def __init__(
    self,
    design_airflow,
    design_external_static_pressure=fr_u(0.5, "in_H2O"),
    design_efficacy=fr_u(0.365,'W/cfm'),
    maximum_power=inf):
      super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
      # Check if design flow is above power limit
      self.maximum_power = maximum_power
      self.EFFICACY_SLOPE_ESP = fr_u(0.235,'(W/cfm)/in_H2O')  # Relative change in efficacy at different external static pressures
      self.SPEED_SLOPE_ESP = fr_u(463.5,'rpm/in_H2O')  # Relative change in rotational speed at different external static pressures
      if self.design_airflow[0]*self.design_efficacy[0] > self.maximum_power:
        pass # TODO: Work backward? Error at first?
      else:
        free_efficacy = self.design_efficacy[0] - self.EFFICACY_SLOPE_ESP*self.design_external_static_pressure
        self.airflow_ratios = [self.design_airflow[i]/self.design_airflow[0] for i in range(self.number_of_speeds)]
        self.free_efficacies = [free_efficacy*self.normalized_free_efficacy(self.airflow_ratios[i]) for i in range(self.number_of_speeds)]

  def normalized_free_efficacy(self, flow_ratio):
    minimum_flow_ratio = 0.293/2.4 # local minima, derived mathematically
    minimum_efficacy = self.free_efficacy_fit(minimum_flow_ratio)
    if flow_ratio < minimum_flow_ratio:
      return minimum_efficacy
    else:
      return self.free_efficacy_fit(flow_ratio)

  @staticmethod
  def free_efficacy_fit(flow_ratio):
    return 0.0981 - 0.293*flow_ratio + 1.2*flow_ratio**2

  def unconstrained_efficacy(self, speed_setting, external_static_pressure):
    return self.free_efficacies[speed_setting] + self.EFFICACY_SLOPE_ESP*external_static_pressure

  def unconstrained_power(self, speed_setting, external_static_pressure):
    return self.design_airflow[speed_setting]*self.unconstrained_efficacy(speed_setting, external_static_pressure)

  def power(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return min(self.unconstrained_power(speed_setting, external_static_pressure), self.maximum_power)

  def airflow(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    if external_static_pressure == 0.:
      return self.design_airflow[speed_setting]
    else:
      estimated_flow_power = self.design_airflow[speed_setting]*external_static_pressure*(self.power(speed_setting, external_static_pressure)/self.unconstrained_power(speed_setting, external_static_pressure))**0.5
      return estimated_flow_power/external_static_pressure

  def efficacy(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return self.power(speed_setting, external_static_pressure)/self.airflow(speed_setting, external_static_pressure)

  def unconstrained_rotational_speed(self, speed_setting, external_static_pressure):
    return (fr_u(1100.,'rpm') - self.SPEED_SLOPE_ESP*(self.design_external_static_pressure - external_static_pressure))*self.airflow_ratios[speed_setting]

  def rotational_speed(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return self.unconstrained_rotational_speed(speed_setting, external_static_pressure)*(self.efficacy(speed_setting, external_static_pressure)/self.unconstrained_efficacy(speed_setting, external_static_pressure))

# TODO: class ECMTorqueFan(Fan)
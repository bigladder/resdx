from .units import fr_u
from math import exp, log

class FanConditions:
  def __init__(self, external_static_pressure=fr_u(0.15, "in_H2O"),
                     speed_setting=0):
    self.external_static_pressure = external_static_pressure
    self.speed_setting = speed_setting

class Fan:
  def __init__(
    self,
    airflow_design,
    external_static_pressure_design,
    efficacy_design=fr_u(0.365,'W/cfm')): # AHRI 210/240 2017 default
      if type(airflow_design) is list:
        self.airflow_design = airflow_design
      else:
        self.airflow_design = [airflow_design]
      self.number_of_speeds = len(self.airflow_design)
      self.external_static_pressure_design = external_static_pressure_design
      if type(efficacy_design) is list:
        self.efficacy_design = efficacy_design
      else:
        self.efficacy_design = [efficacy_design]*self.number_of_speeds

      self.system_exponent = 0.5
      self.system_curve_constant = self.external_static_pressure_design**(self.system_exponent)/self.airflow_design[0]

    # system curve function: given flow rate, calculate pressure
  def system_pressure(self, airflow):
    (airflow*self.system_curve_constant)**(1./self.system_exponent)

  def system_flow(self, pressure):
    pressure**(self.system_exponent)/self.airflow_design[0]

  def efficacy(self, conditions):
    raise NotImplementedError()

  def airflow(self, conditions):
    raise NotImplementedError()

  def power(self, conditions):
    return self.airflow(conditions)*self.efficacy(conditions)

  def rotational_speed(self, conditions):
    raise NotImplementedError()

  def write_A205(self):
    pass

class ConstantEfficacyFan(Fan):
  def efficacy(self, conditions):
    return self.efficacy_design[conditions.speed_setting]

  def airflow(self, conditions):
    return self.airflow_design[conditions.speed_setting]

class PSCFan(Fan):
  '''Based largely on measured fan performance by Proctor Engineering'''
  def __init__(
    self,
    airflow_design,
    external_static_pressure_design=fr_u(0.15, "in_H2O"),
    efficacy_design=fr_u(0.365,'W/cfm')):
      super().__init__(airflow_design, external_static_pressure_design, efficacy_design)
      self.AIRFLOW_COEFFICIENT = fr_u(10.,'cfm')
      self.AIRFLOW_EXP_COEFFICIENT = fr_u(5.355391179,'1/in_H2O')
      self.EFFICACY_SLOPE = -0.1446674009  # Relative change in efficacy at lower flow ratios
      airflow_reduction_design = self.airflow_reduction(external_static_pressure_design)
      self.airflow_free = [self.airflow_design[i] + airflow_reduction_design for i in range(self.number_of_speeds)]
      self.airflow_ratios = [self.airflow_free[i]/self.airflow_free[0] for i in range(self.number_of_speeds)]
      self.efficacy_design = efficacy_design
      self.efficacies = [efficacy_design*(1. + self.EFFICACY_SLOPE*(self.airflow_ratios[i] - 1.)) for i in range(self.number_of_speeds)]
      self.block_pressure = [log(self.airflow_free[i]/self.AIRFLOW_COEFFICIENT + 1.)/self.AIRFLOW_EXP_COEFFICIENT for i in range(self.number_of_speeds)]
      self.speed_free = [fr_u(1040.,'rpm')*self.airflow_ratios[i] for i in range(self.number_of_speeds)]

  def efficacy(self, conditions):
      return self.efficacies[conditions.speed_setting]

  def airflow(self, conditions):
    i = conditions.speed_setting
    return max(self.airflow_free[i] - self.airflow_reduction(conditions.external_static_pressure),0.)

  def rotational_speed(self, conditions):
    i = conditions.speed_setting
    if conditions.external_static_pressure > self.block_pressure[i]:
      return fr_u(1100.,'rpm')
    else:
      return self.speed_free[i] + (fr_u(1100.,'rpm') - self.speed_free[i])*conditions.external_static_pressure/self.block_pressure[i]

  def airflow_reduction(self, external_static_pressure):
    return self.AIRFLOW_COEFFICIENT*(exp(external_static_pressure*self.AIRFLOW_EXP_COEFFICIENT)-1.)

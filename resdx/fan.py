from .units import fr_u
from math import exp, log

class FanConditions:
  def __init__(self, external_static_pressure=fr_u(0.15, "in_H2O"),
                     speed_setting=0):
    self.external_static_pressure = external_static_pressure
    self.speed_setting = speed_setting

class Fan:
  '''Default fan behavior is similar to the default assumptions in AHRI 210/240'''
  def __init__(
    self,
    airflow_rated,
    external_static_pressure_rated,
    efficacy_rated):
      if type(airflow_rated) is list:
        self.airflow_rated = airflow_rated
      else:
        self.airflow_rated = [airflow_rated]
      self.number_of_speeds = len(self.airflow_rated)

  def efficacy(self, conditions):
    min_esp_2017 = fr_u(0.2, "in_H2O")
    min_esp_2021 = fr_u(0.50, "in_H2O")
    weight = (conditions.external_static_pressure - min_esp_2017) / (min_esp_2021 - min_esp_2017)
    weight = max(min(1., weight),0.)
    return fr_u(0.365,'W/cfm')*(1. - weight) + fr_u(0.441,'W/cfm')*weight

  def airflow(self, conditions):
    return self.rated_airflow

  def power(self, conditions):
    return self.airflow(conditions)*self.efficacy(conditions)

  def rotational_speed(self, conditions):
    raise NotImplementedError()

  def write_A205(self):
    pass

class PSCFan(Fan):
  '''Based largely on measured fan performance by Proctor Engineering'''
  def __init__(
    self,
    airflow_rated,
    external_static_pressure_rated=fr_u(0.15, "in_H2O"),
    efficacy_rated=fr_u(0.365,'W/cfm')):
      super().__init__(airflow_rated, external_static_pressure_rated, efficacy_rated)
      self.AIRFLOW_COEFFICIENT = fr_u(10.,'cfm')
      self.AIRFLOW_EXP_COEFFICIENT = fr_u(5.355391179,'1/in_H2O')
      self.EFFICACY_SLOPE = -0.1446674009  # Relative change in efficacy at lower flow ratios
      airflow_reduction_rated = self.airflow_reduction(external_static_pressure_rated)
      self.airflow_free = [self.airflow_rated[i] + airflow_reduction_rated for i in range(self.number_of_speeds)]
      self.airflow_ratios = [self.airflow_free[i]/self.airflow_free[0] for i in range(self.number_of_speeds)]
      self.efficacy_rated = efficacy_rated
      self.efficacies = [efficacy_rated*(1. + self.EFFICACY_SLOPE*(self.airflow_ratios[i] - 1.)) for i in range(self.number_of_speeds)]
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

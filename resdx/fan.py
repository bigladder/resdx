from koozie import fr_u, to_u, convert
from math import exp, log, inf
from scipy import optimize # Used for finding system/fan curve intersection
from numpy import linspace, array
import uuid
import datetime
from random import Random

class FanMetadata:
  def __init__(
    self,
    description="",
    data_source="https://github.com/bigladder/resdx",
    notes="",
    uuid_seed=None
    ):

    self.description = description
    self.data_source = data_source
    self.notes = notes
    self.uuid_seed = uuid_seed

class Fan:
  def __init__(self,
               design_airflow,
               design_external_static_pressure,
               design_efficacy=fr_u(0.365,'W/cfm'), # AHRI 210/240 2017 default
               metadata=None):
    self.design_external_static_pressure = design_external_static_pressure
    self.design_efficacy = design_efficacy
    self.number_of_speeds = 0
    self.design_airflow = []
    self.design_airflow_ratio = []

    if metadata is None:
      self.metadata = FanMetadata()
    else:
      self.metadata = metadata

    if type(design_airflow) is list:
      for airflow in design_airflow:
        self.add_speed(airflow)
    else:
      self.add_speed(design_airflow)
    self.system_exponent = 0.5
    self.system_curve_constant = self.design_external_static_pressure**(self.system_exponent)/self.design_airflow[0]

  def add_speed(self, airflow, external_static_pressure=None):
    self.design_airflow.append(airflow)
    self.number_of_speeds += 1
    self.design_airflow_ratio.append(self.design_airflow[-1]/self.design_airflow[0])

  def remove_speed(self, speed_setting):
    self.design_airflow.pop(speed_setting)
    self.number_of_speeds -= 1
    self.design_airflow_ratio.pop(speed_setting)

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

  def check_power(self, airflow, external_static_pressure=None):
    self.add_speed(airflow)
    new_speed_setting = self.number_of_speeds - 1
    power = self.power(new_speed_setting, external_static_pressure)
    self.remove_speed(new_speed_setting)
    return power

  def find_rated_fan_speed(
    self,
    gross_capacity,
    rated_flow_per_rated_net_capacity,
    guess_airflow=None,
    rated_full_flow_external_static_pressure=None,
    cooling=True):
    '''Given a gross capacity, and the rated flow per rated net capacity, find the speed and flow rate that gives consistent results'''
    '''Q_gross +/- Q_fan = Q_net'''
    if guess_airflow is None:
      guess_airflow = gross_capacity*rated_flow_per_rated_net_capacity

    if rated_full_flow_external_static_pressure is not None:
      full_airflow = self.airflow(0, rated_full_flow_external_static_pressure)
      pressure_function = lambda x : rated_full_flow_external_static_pressure*(x/full_airflow)**2
    else:
      pressure_function = lambda x : None

    if cooling:
      net_capacity_function = lambda x : gross_capacity - self.check_power(x, pressure_function(x))
    else:
      net_capacity_function = lambda x : gross_capacity + self.check_power(x, pressure_function(x))

    root_fn = lambda x : net_capacity_function(x) - x/rated_flow_per_rated_net_capacity
    f, solution = optimize.newton(root_fn, guess_airflow, full_output = True)
    self.add_speed(f, pressure_function(f))
    return f

  def get_speed_order_map(self):
    airflow_list = array([self.airflow(speed, 0.) for speed in range(self.number_of_speeds)])
    return airflow_list.argsort()

  def generate_205_representation(self):
    timestamp = datetime.datetime.now().isoformat("T","minutes")
    rnd = Random()
    if self.metadata.uuid_seed is None:
      self.metadata.uuid_seed = hash(self)
    rnd.seed(self.metadata.uuid_seed)
    unique_id = str(uuid.UUID(int=rnd.getrandbits(128), version=4))

    speed_order_map = self.get_speed_order_map()
    max_speed = speed_order_map[-1]

    if type(self) is ECMFlowFan:
      if self.maximum_power is inf:
        # Assumption
        max_power = self.power(speed_setting=max_speed,external_static_pressure=fr_u(1.2,"in_H2O"))*1.2
      else:
        max_power = self.maximum_power
    else:
      # Assumption
      max_power = self.power(speed_setting=max_speed,external_static_pressure=0.)*1.2

    # RS0005 Motor
    rnd.seed(unique_id)

    metadata_motor = {
      "data_model": "ASHRAE_205",
      "schema": "RS0005",
      "schema_version": "1.0.0",
      "description": f"Placeholder motor representation (performance characterized in parent RS0003 fan assembly)",
      "id": str(uuid.UUID(int=rnd.getrandbits(128), version=4)),
      "data_timestamp": f"{timestamp}Z",
      "data_version": 1,
      "data_source": self.metadata.data_source,
      "disclaimer": "This data is synthetic and does not represent any physical products.",
    }

    performance_motor = {
      "maximum_power": max_power,
      "standby_power": 0.,
      "number_of_poles": 6,
    }

    design_airflow = self.design_airflow[0] if type(self.design_airflow) is list else self.design_airflow
    design_efficacy = self.design_efficacy[0] if type(self.design_efficacy) is list else self.design_efficacy
    design_external_static_pressure = self.design_external_static_pressure[0] if type(self.design_external_static_pressure) is list else self.design_external_static_pressure

    if len(self.metadata.description) == 0:
      airflow_cfm = to_u(design_airflow,'cfm')
      efficacy_w_cfm = to_u(design_efficacy,'W/cfm')
      pressure_in_h2o = to_u(design_external_static_pressure,'in_H2O')
      if type(self) is PSCFan:
        fan_description = "Permanent Split Capacitor (PSC) fan"
      elif type(self) is ECMFlowFan:
        fan_description = f"Electronically Commutated Motor (ECM) fan"
      else:
        fan_description = "fan"
      self.metadata.description = f"{airflow_cfm:.0f} cfm {fan_description} ({efficacy_w_cfm:.3f} W/cfm @ {pressure_in_h2o:.2f} in. H2O)"

    metadata = {
      "data_model": "ASHRAE_205",
      "schema": "RS0003",
      "schema_version": "1.0.0",
      "description": self.metadata.description,
      "id": unique_id,
      "data_timestamp": f"{timestamp}Z",
      "data_version": 1,
      "data_source": self.metadata.data_source,
      "disclaimer": "This data is synthetic and does not represent any physical products."
    }

    if len(self.metadata.notes) > 0:
      metadata["notes"] = self.metadata.notes

    # Create conditions
    speed_number = list(range(1,self.number_of_speeds + 1))

    if type(self) is PSCFan:
      max_static_pressure = self.block_pressure[max_speed]
    else:
      max_static_pressure = fr_u(1.2, "in_H2O")
    static_pressure_difference = linspace(fr_u(0., "in_H2O"), max_static_pressure, 6).tolist()

    grid_variables = {
      "speed_number": speed_number,
      "static_pressure_difference": static_pressure_difference
    }

    standard_air_volumetric_flow_rate = []
    shaft_power = []
    impeller_rotational_speed = []

    for speed in speed_number:
      for esp in static_pressure_difference:
        # Get speed number from ordered number
        positional_speed_number = speed_order_map[speed - 1]
        shaft_power.append(self.power(speed_setting=positional_speed_number,external_static_pressure=esp))
        standard_air_volumetric_flow_rate.append(self.airflow(speed_setting=positional_speed_number,external_static_pressure=esp))
        impeller_rotational_speed.append(to_u(self.rotational_speed(speed_setting=positional_speed_number,external_static_pressure=esp),"rps"))

    performance_map = {
      "grid_variables": grid_variables,
      "lookup_variables": {
        "shaft_power": shaft_power,
        "standard_air_volumetric_flow_rate": standard_air_volumetric_flow_rate,
        "impeller_rotational_speed": impeller_rotational_speed
      }
    }

    performance = {
      "nominal_standard_air_volumetric_flow_rate": self.airflow(speed_setting=max_speed,external_static_pressure=self.design_external_static_pressure),
      "is_enclosed": True,
      "assembly_components": [
        {
          "component_type": "COIL",
          "wet_pressure_difference": 75. #Pa
        }],
        "heat_loss_fraction": 1.,
        "maximum_impeller_rotational_speed": convert(1500.,"rpm","rps"),
        "minimum_impeller_rotational_speed": 0.,
        "operation_speed_control_type": "DISCRETE",
        "installation_speed_control_type": "FIXED",
        "motor_representation": {"metadata": metadata_motor, "performance": performance_motor},
        "performance_map": performance_map

    }

    representation = {"metadata": metadata, "performance": performance}

    return representation

class ConstantEfficacyFan(Fan):
  def __init__(self, design_airflow, design_external_static_pressure, design_efficacy=fr_u(0.365, 'W/cfm')):
    super().__init__(design_airflow, design_external_static_pressure, design_efficacy)
    if type(self.design_efficacy) is not list:
      self.design_efficacy = [self.design_efficacy]*self.number_of_speeds

  def add_speed(self, airflow, efficacy=None, external_static_pressure=None):
    super().add_speed(airflow, external_static_pressure)
    if efficacy is not None:
      self.design_efficacy.append(efficacy)

  def remove_speed(self, speed_setting):
    super().remove_speed(speed_setting)
    self.design_efficacy.pop(speed_setting)

  def efficacy(self, speed_setting, external_static_pressure=None):
    return self.design_efficacy[speed_setting]

  def airflow(self, speed_setting, external_static_pressure=None):
    return self.design_airflow[speed_setting]

class PSCFan(Fan):
  '''Based largely on measured fan performance by Proctor Engineering'''
  '''Model needs more data to refine and further generalize'''

  AIRFLOW_COEFFICIENT = fr_u(10.,'cfm')
  AIRFLOW_EXP_COEFFICIENT = fr_u(5.35,'1/in_H2O')
  EFFICACY_SLOPE = 0.3  # Relative change in efficacy at lower flow ratios (data is fairly inconsistent on this value)

  def __init__(
    self,
    design_airflow,
    design_external_static_pressure=fr_u(0.5, "in_H2O"),
    design_efficacy=fr_u(0.365,'W/cfm')):
      self.design_airflow_reduction = self.airflow_reduction(design_external_static_pressure)
      self.free_airflow = []
      self.free_airflow_ratio = []
      self.speed_efficacy = []
      self.block_pressure = []
      self.free_speed = []
      super().__init__(design_airflow, design_external_static_pressure, design_efficacy)

  def add_speed(self, airflow, external_static_pressure=None):
    if external_static_pressure is not None:
      design_airflow = airflow - (self.design_airflow_reduction - self.airflow_reduction(external_static_pressure))
    else:
      design_airflow = airflow
    super().add_speed(design_airflow)
    self.free_airflow.append(self.design_airflow[-1] + self.design_airflow_reduction)
    self.free_airflow_ratio.append(self.free_airflow[-1]/self.free_airflow[0])
    self.speed_efficacy.append(self.design_efficacy*(1. + self.EFFICACY_SLOPE*(self.free_airflow_ratio[-1] - 1.)))
    self.block_pressure.append(log(self.free_airflow[-1]/self.AIRFLOW_COEFFICIENT + 1.)/self.AIRFLOW_EXP_COEFFICIENT)
    self.free_speed.append(fr_u(1040.,'rpm')*self.free_airflow_ratio[-1])

  def remove_speed(self, speed_setting):
    super().remove_speed(speed_setting)
    self.free_airflow.pop(speed_setting)
    self.free_airflow_ratio.pop(speed_setting)
    self.speed_efficacy.pop(speed_setting)
    self.block_pressure.pop(speed_setting)
    self.free_speed.pop(speed_setting)

  def efficacy(self, speed_setting, external_static_pressure=None):
      return self.speed_efficacy[speed_setting]

  def airflow(self, speed_setting, external_static_pressure=None):
    i = speed_setting
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return max(self.free_airflow[speed_setting] - self.airflow_reduction(external_static_pressure),0.)

  def rotational_speed(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    if external_static_pressure > self.block_pressure[speed_setting]:
      return fr_u(1100.,'rpm')
    else:
      i = speed_setting
      return self.free_speed[i] + (fr_u(1100.,'rpm') - self.free_speed[i])*external_static_pressure/self.block_pressure[i]

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

  EFFICACY_SLOPE_ESP = fr_u(0.235,'(W/cfm)/in_H2O')  # Relative change in efficacy at different external static pressures
  SPEED_SLOPE_ESP = fr_u(463.5,'rpm/in_H2O')  # Relative change in rotational speed at different external static pressures

  def __init__(
    self,
    design_airflow,
    design_external_static_pressure=fr_u(0.5, "in_H2O"),
    design_efficacy=fr_u(0.365,'W/cfm'),
    maximum_power=inf):
      # Check if design power is above power limit
      design_power = (design_airflow[0] if type(design_airflow) is list else design_airflow)*design_efficacy
      if design_power > maximum_power:
        raise Exception(f"Design power ({design_power} W) is greater than the maximum power ({maximum_power}) W")
      self.maximum_power = maximum_power
      self.design_free_efficacy = design_efficacy - self.EFFICACY_SLOPE_ESP*design_external_static_pressure
      self.free_efficacy = []
      super().__init__(design_airflow, design_external_static_pressure, design_efficacy)

  def add_speed(self, airflow, external_static_pressure=None):
    super().add_speed(airflow, external_static_pressure)
    self.free_efficacy.append(self.design_free_efficacy*self.normalized_free_efficacy(self.design_airflow_ratio[-1]))

  def remove_speed(self, speed_setting):
    super().remove_speed(speed_setting)
    self.free_efficacy.pop(speed_setting)

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
    return self.free_efficacy[speed_setting] + self.EFFICACY_SLOPE_ESP*external_static_pressure

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
    return (fr_u(1100.,'rpm') - self.SPEED_SLOPE_ESP*(self.design_external_static_pressure - external_static_pressure))*self.design_airflow_ratio[speed_setting]

  def rotational_speed(self, speed_setting, external_static_pressure=None):
    if external_static_pressure is None:
      external_static_pressure = self.operating_pressure(speed_setting)
    return self.unconstrained_rotational_speed(speed_setting, external_static_pressure)*(self.efficacy(speed_setting, external_static_pressure)/self.unconstrained_efficacy(speed_setting, external_static_pressure))

# TODO: class ECMTorqueFan(Fan)

import copy
import types

from resdx.conditions import CoolingConditions, HeatingConditions
from resdx.psychrometrics import PsychState
from koozie import fr_u
from .dx_unit import DXUnit
from scipy import interpolate
from .models import RESNETDXModel
from .fan import ConstantEfficacyFan

class VCHPDataPoint:
  def __init__(self,drybulb,capacities,cops):
      # capacities and cops are array like objects with one value for each capacity stage
      self.drybulb = drybulb
      self.capacities = capacities
      self.cops = cops

class VCHPDataPoints(list):

  def setup(self):
    # Sort arrays with increasing drybulb
    self.sort(key=lambda point : point.drybulb)

    # TODO: Check to make sure all arrays are the same length

    self.number_of_stages = len(self[0].capacities)
    self.get_capacity = [None]*self.number_of_stages
    self.get_cop = [None]*self.number_of_stages

    # create arrays of functions for interpolation: 0 = @ maximum capacity, 1 = @ minimum capacity
    for i in range(self.number_of_stages):
      self.get_capacity[i] = interpolate.interp1d(
        [point.drybulb for point in self],
        [point.capacities[i] for point in self],
        fill_value="extrapolate")
      self.get_cop[i] = interpolate.interp1d(
        [point.drybulb for point in self],
        [point.cops[i] for point in self],
        fill_value="extrapolate")


def make_vchp_unit(
  net_cooling_data,
  net_heating_data,
  cooling_full_load_speed_ratio=1.0,
  cooling_intermediate_stage_speed_ratio=0.5,
  heating_full_load_speed_ratio=1.0,
  heating_intermediate_stage_speed_ratio=0.5,
  c_d_cooling=0.25,
  c_d_heating=0.25,
  rated_cooling_fan_efficacy=[fr_u(0.2075,'W/cfm')]*2,
  rated_heating_fan_efficacy=[fr_u(0.2075,'W/cfm')]*2,
  rated_cooling_airflow_per_rated_net_capacity=[fr_u(400.0,"cfm/ton_ref")]*2,
  rated_heating_airflow_per_rated_net_capacity=[fr_u(400.0,"cfm/ton_ref")]*2,
  base_model=RESNETDXModel()):

  max_cooling_speed = 0
  min_cooling_speed = 1

  max_heating_speed = 0
  min_heating_speed = 1

  # Add "full" speed
  if cooling_full_load_speed_ratio < 1.0:
    full_cooling_speed = 1
    for point in net_cooling_data:
      point.capacities.insert(full_cooling_speed, point.capacities[min_cooling_speed] + cooling_full_load_speed_ratio*(point.capacities[max_cooling_speed] - point.capacities[min_cooling_speed]))
      point.cops.insert(full_cooling_speed, point.cops[min_cooling_speed] + cooling_full_load_speed_ratio*(point.cops[max_cooling_speed] - point.cops[min_cooling_speed]))
    rated_cooling_fan_efficacy.insert(full_cooling_speed, rated_cooling_fan_efficacy[min_cooling_speed] + cooling_full_load_speed_ratio*(rated_cooling_fan_efficacy[max_cooling_speed] - rated_cooling_fan_efficacy[min_cooling_speed]))
    rated_cooling_airflow_per_rated_net_capacity.insert(full_cooling_speed, rated_cooling_airflow_per_rated_net_capacity[min_cooling_speed] + cooling_full_load_speed_ratio*(rated_cooling_airflow_per_rated_net_capacity[max_cooling_speed] - rated_cooling_airflow_per_rated_net_capacity[min_cooling_speed]))
    min_cooling_speed = full_cooling_speed + 1
  else:
    full_cooling_speed = 0

  if heating_full_load_speed_ratio < 1.0:
    full_heating_speed = 1
    for point in net_heating_data:
      point.capacities.insert(full_heating_speed, point.capacities[min_heating_speed] + heating_full_load_speed_ratio*(point.capacities[max_heating_speed] - point.capacities[min_heating_speed]))
      point.cops.insert(full_heating_speed, point.cops[min_heating_speed] + heating_full_load_speed_ratio*(point.cops[max_heating_speed] - point.cops[min_heating_speed]))
    rated_heating_fan_efficacy.insert(full_heating_speed, rated_heating_fan_efficacy[min_heating_speed] + heating_full_load_speed_ratio*(rated_heating_fan_efficacy[max_heating_speed] - rated_heating_fan_efficacy[min_heating_speed]))
    rated_heating_airflow_per_rated_net_capacity.insert(full_heating_speed, rated_heating_airflow_per_rated_net_capacity[min_heating_speed] + heating_full_load_speed_ratio*(rated_heating_airflow_per_rated_net_capacity[max_heating_speed] - rated_heating_airflow_per_rated_net_capacity[min_heating_speed]))
    min_heating_speed = full_heating_speed + 1
  else:
    full_heating_speed = 0

  # Add intermediate speed
  intermediate_cooling_speed = full_cooling_speed + 1
  for point in net_cooling_data:
    point.capacities.insert(intermediate_cooling_speed, point.capacities[min_cooling_speed] + cooling_intermediate_stage_speed_ratio*(point.capacities[full_cooling_speed] - point.capacities[min_cooling_speed]))
    point.cops.insert(intermediate_cooling_speed, point.cops[min_cooling_speed] + cooling_intermediate_stage_speed_ratio*(point.cops[full_cooling_speed] - point.cops[min_cooling_speed]))
  rated_cooling_fan_efficacy.insert(intermediate_cooling_speed, rated_cooling_fan_efficacy[min_cooling_speed] + cooling_intermediate_stage_speed_ratio*(rated_cooling_fan_efficacy[full_cooling_speed] - rated_cooling_fan_efficacy[min_cooling_speed]))
  rated_cooling_airflow_per_rated_net_capacity.insert(intermediate_cooling_speed, rated_cooling_airflow_per_rated_net_capacity[min_cooling_speed] + cooling_intermediate_stage_speed_ratio*(rated_cooling_airflow_per_rated_net_capacity[full_cooling_speed] - rated_cooling_airflow_per_rated_net_capacity[min_cooling_speed]))
  min_cooling_speed = intermediate_cooling_speed + 1

  intermediate_heating_speed = full_heating_speed + 1
  for point in net_heating_data:
    point.capacities.insert(intermediate_heating_speed, point.capacities[min_heating_speed] + heating_intermediate_stage_speed_ratio*(point.capacities[full_heating_speed] - point.capacities[min_heating_speed]))
    point.cops.insert(intermediate_heating_speed, point.cops[min_heating_speed] + heating_intermediate_stage_speed_ratio*(point.cops[full_heating_speed] - point.cops[min_heating_speed]))
  rated_heating_fan_efficacy.insert(intermediate_heating_speed, rated_heating_fan_efficacy[min_heating_speed] + heating_intermediate_stage_speed_ratio*(rated_heating_fan_efficacy[full_heating_speed] - rated_heating_fan_efficacy[min_heating_speed]))
  rated_heating_airflow_per_rated_net_capacity.insert(intermediate_heating_speed, rated_heating_airflow_per_rated_net_capacity[min_heating_speed] + heating_intermediate_stage_speed_ratio*(rated_heating_airflow_per_rated_net_capacity[full_heating_speed] - rated_heating_airflow_per_rated_net_capacity[min_heating_speed]))
  min_heating_speed = intermediate_heating_speed + 1

  # Setup data
  net_cooling_data.setup()
  net_heating_data.setup()

  net_total_cooling_capacity = [net_cooling_data.get_capacity[i](fr_u(95.0,"°F")) for i in range(net_cooling_data.number_of_stages)]
  cooling_rated_fan_power = [net_total_cooling_capacity[i]*rated_cooling_fan_efficacy[i]*rated_cooling_airflow_per_rated_net_capacity[i] for i in range(net_cooling_data.number_of_stages)]
  net_heating_capacity = [net_heating_data.get_capacity[i](fr_u(47.0,"°F")) for i in range(net_heating_data.number_of_stages)]
  heating_rated_fan_power = [net_heating_capacity[i]*rated_heating_fan_efficacy[i]*rated_heating_airflow_per_rated_net_capacity[i] for i in range(net_heating_data.number_of_stages)]

  gross_cooling_data = copy.deepcopy(net_cooling_data)
  for point in gross_cooling_data:
    for i in range(gross_cooling_data.number_of_stages):
      net_power = point.capacities[i]/point.cops[i]
      gross_capacity = point.capacities[i] + cooling_rated_fan_power[i] # removing fan heat increases capacity
      gross_power = net_power - cooling_rated_fan_power[i]
      point.capacities[i] = gross_capacity
      point.cops[i] = gross_capacity/gross_power

  gross_heating_data = copy.deepcopy(net_heating_data)
  for point in gross_heating_data:
    for i in range(gross_heating_data.number_of_stages):
      net_power = point.capacities[i]/point.cops[i]
      gross_capacity = point.capacities[i] - heating_rated_fan_power[i] # removing fan heat decreases capacity
      gross_power = net_power - heating_rated_fan_power[i]
      point.capacities[i] = gross_capacity
      point.cops[i] = gross_capacity/gross_power

  gross_cooling_data.setup()
  gross_heating_data.setup()

  # Setup fan
  airflows = []
  efficacies = []
  fan_speed = 0
  cooling_fan_speed = []
  heating_fan_speed = []
  for i in range(gross_cooling_data.number_of_stages):
    airflows.append(net_total_cooling_capacity[i]*rated_cooling_airflow_per_rated_net_capacity[i])
    efficacies.append(rated_cooling_fan_efficacy[i])
    cooling_fan_speed.append(fan_speed)
    fan_speed += 1
  for i in range(gross_heating_data.number_of_stages):
    airflows.append(net_heating_capacity[i]*rated_heating_airflow_per_rated_net_capacity[i])
    efficacies.append(rated_heating_fan_efficacy[i])
    heating_fan_speed.append(fan_speed)
    fan_speed += 1

  fan = ConstantEfficacyFan(airflows, fr_u(0.50, "in_H2O"), design_efficacy=efficacies)

  new_model = copy.deepcopy(base_model)

  def new_gross_cooling_power(self, conditions):
    rated_conditions = self.system.make_condition(CoolingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_cooling_power(single_speed_conditions) / base_model.gross_cooling_power(rated_conditions)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db) / gross_cooling_data.get_cop[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_total_cooling_capacity(self, conditions):

    rated_conditions = self.system.make_condition(CoolingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_total_cooling_capacity(single_speed_conditions) / base_model.gross_total_cooling_capacity(rated_conditions)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_steady_state_heating_capacity(self, conditions):
    rated_conditions = self.system.make_condition(HeatingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_steady_state_heating_capacity(single_speed_conditions) / base_model.gross_steady_state_heating_capacity(rated_conditions)

    return correction_factor * gross_heating_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_steady_state_heating_power(self, conditions):
    rated_conditions = self.system.make_condition(HeatingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_steady_state_heating_power(single_speed_conditions) / base_model.gross_steady_state_heating_power(rated_conditions)

    return correction_factor * gross_heating_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db) / gross_heating_data.get_cop[conditions.compressor_speed](conditions.outdoor.db)

  new_model.gross_cooling_power = types.MethodType(new_gross_cooling_power, new_model)
  new_model.gross_total_cooling_capacity = types.MethodType(new_gross_total_cooling_capacity, new_model)
  new_model.gross_steady_state_heating_capacity = types.MethodType(new_gross_steady_state_heating_capacity, new_model)
  new_model.gross_steady_state_heating_power = types.MethodType(new_gross_steady_state_heating_power, new_model)

  return DXUnit(model=new_model,
    rated_net_total_cooling_capacity = net_total_cooling_capacity,
    rated_net_cooling_cop = [net_cooling_data.get_cop[i](fr_u(95.0,"°F")) for i in range(net_cooling_data.number_of_stages)],
    rated_gross_cooling_cop  =  None, # Use net instead
    c_d_cooling = c_d_cooling,
    rated_net_heating_capacity = net_heating_capacity,
    rated_net_heating_cop = [net_heating_data.get_cop[i](fr_u(47.0,"°F")) for i in range(net_heating_data.number_of_stages)],
    rated_gross_heating_cop  =  None, # Use net instead
    c_d_heating = c_d_heating,
    fan = fan,
    heating_fan_speed = heating_fan_speed,
    cooling_fan_speed = cooling_fan_speed,
    rated_heating_fan_speed = heating_fan_speed,
    rated_cooling_fan_speed = cooling_fan_speed,
    cooling_full_load_speed = 1 if net_cooling_data.number_of_stages > 3 else 0,
    cooling_intermediate_speed = net_cooling_data.number_of_stages - 2,
    heating_full_load_speed = 1 if net_heating_data.number_of_stages > 3 else 0,
    heating_intermediate_speed = net_heating_data.number_of_stages - 2,
    base_model=base_model
    )

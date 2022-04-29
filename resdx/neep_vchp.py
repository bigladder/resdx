import copy

from resdx.conditions import CoolingConditions, HeatingConditions
from resdx.psychrometrics import PsychState
from .units import fr_u, to_u
from .dx_unit import DXUnit
from scipy import interpolate
from .models import RESNETDXModel

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
  fan_efficacy_cooling_rated=[fr_u(0.2075,'W/(cu_ft/min)')]*2,
  fan_efficacy_heating_rated=[fr_u(0.2075,'W/(cu_ft/min)')]*2,
  flow_rated_per_cap_cooling_rated=[fr_u(400.0,"(cu_ft/min)/ton_of_refrigeration")]*2,
  flow_rated_per_cap_heating_rated=[fr_u(400.0,"(cu_ft/min)/ton_of_refrigeration")]*2,
  base_model=RESNETDXModel()):

  # Add "full" speed
  if cooling_full_load_speed_ratio < 1.0:
    for point in net_cooling_data:
      point.capacities.insert(1, point.capacities[1] + cooling_full_load_speed_ratio*(point.capacities[0] - point.capacities[1]))
      point.cops.insert(1, point.cops[1] + cooling_full_load_speed_ratio*(point.cops[0] - point.cops[1]))
    fan_efficacy_cooling_rated.insert(1, fan_efficacy_cooling_rated[1] + cooling_full_load_speed_ratio*(fan_efficacy_cooling_rated[0] - fan_efficacy_cooling_rated[1]))
    flow_rated_per_cap_cooling_rated.insert(1, flow_rated_per_cap_cooling_rated[1] + cooling_full_load_speed_ratio*(flow_rated_per_cap_cooling_rated[0] - flow_rated_per_cap_cooling_rated[1]))

  if heating_full_load_speed_ratio < 1.0:
    for point in net_heating_data:
      point.capacities.insert(1, point.capacities[1] + heating_full_load_speed_ratio*(point.capacities[0] - point.capacities[1]))
      point.cops.insert(1, point.cops[1] + heating_full_load_speed_ratio*(point.cops[0] - point.cops[1]))
    fan_efficacy_heating_rated.insert(1, fan_efficacy_heating_rated[1] + heating_full_load_speed_ratio*(fan_efficacy_heating_rated[0] - fan_efficacy_heating_rated[1]))
    flow_rated_per_cap_heating_rated.insert(1, flow_rated_per_cap_heating_rated[1] + heating_full_load_speed_ratio*(flow_rated_per_cap_heating_rated[0] - flow_rated_per_cap_heating_rated[1]))

  # Add intermediate speed
  for point in net_cooling_data:
    point.capacities.insert(1, point.capacities[1] + cooling_intermediate_stage_speed_ratio*(point.capacities[0] - point.capacities[1]))
    point.cops.insert(1, point.cops[1] + cooling_intermediate_stage_speed_ratio*(point.cops[0] - point.cops[1]))
  fan_efficacy_cooling_rated.insert(1, fan_efficacy_cooling_rated[1] + cooling_intermediate_stage_speed_ratio*(fan_efficacy_cooling_rated[0] - fan_efficacy_cooling_rated[1]))
  flow_rated_per_cap_cooling_rated.insert(1, flow_rated_per_cap_cooling_rated[1] + cooling_intermediate_stage_speed_ratio*(flow_rated_per_cap_cooling_rated[0] - flow_rated_per_cap_cooling_rated[1]))

  for point in net_heating_data:
    point.capacities.insert(1, point.capacities[1] + heating_intermediate_stage_speed_ratio*(point.capacities[0] - point.capacities[1]))
    point.cops.insert(1, point.cops[1] + heating_intermediate_stage_speed_ratio*(point.cops[0] - point.cops[1]))
  fan_efficacy_heating_rated.insert(1, fan_efficacy_heating_rated[1] + heating_intermediate_stage_speed_ratio*(fan_efficacy_heating_rated[0] - fan_efficacy_heating_rated[1]))
  flow_rated_per_cap_heating_rated.insert(1, flow_rated_per_cap_heating_rated[1] + heating_intermediate_stage_speed_ratio*(flow_rated_per_cap_heating_rated[0] - flow_rated_per_cap_heating_rated[1]))

  # Setup data
  net_cooling_data.setup()
  net_heating_data.setup()

  cooling_rated_fan_power = [net_cooling_data.get_capacity[i](fr_u(95.0,"°F"))*fan_efficacy_cooling_rated[i]*flow_rated_per_cap_cooling_rated[i] for i in range(net_cooling_data.number_of_stages)]
  heating_rated_fan_power = [net_heating_data.get_capacity[i](fr_u(47.0,"°F"))*fan_efficacy_heating_rated[i]*flow_rated_per_cap_heating_rated[i] for i in range(net_heating_data.number_of_stages)]

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

  new_model = copy.deepcopy(base_model)

  def new_gross_cooling_power(conditions, system):
    rated_conditions = system.make_condition(CoolingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_cooling_power(single_speed_conditions, system) / base_model.gross_cooling_power(rated_conditions, system)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db) / gross_cooling_data.get_cop[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_total_cooling_capacity(conditions, system):
    rated_conditions = system.make_condition(CoolingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_total_cooling_capacity(single_speed_conditions, system) / base_model.gross_total_cooling_capacity(rated_conditions, system)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_steady_state_heating_capacity(conditions, system):
    rated_conditions = system.make_condition(HeatingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_steady_state_heating_capacity(single_speed_conditions, system) / base_model.gross_steady_state_heating_capacity(rated_conditions, system)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db)

  def new_gross_steady_state_heating_power(conditions, system):
    rated_conditions = system.make_condition(HeatingConditions,compressor_speed=0,outdoor=PsychState(drybulb=conditions.outdoor.db,rel_hum=0.4))

    single_speed_conditions = copy.deepcopy(conditions)
    single_speed_conditions.compressor_speed = 0

    correction_factor = base_model.gross_steady_state_heating_power(single_speed_conditions, system) / base_model.gross_steady_state_heating_power(rated_conditions, system)

    return correction_factor * gross_cooling_data.get_capacity[conditions.compressor_speed](conditions.outdoor.db) / gross_cooling_data.get_cop[conditions.compressor_speed](conditions.outdoor.db)

  new_model.gross_cooling_power = new_gross_cooling_power
  new_model.gross_total_cooling_capacity = new_gross_total_cooling_capacity
  new_model.gross_steady_state_heating_capacity = new_gross_steady_state_heating_capacity
  new_model.gross_steady_state_heating_power = new_gross_steady_state_heating_power

  return DXUnit(model=new_model,
    net_total_cooling_capacity_rated = [net_cooling_data.get_capacity[i](fr_u(95.0,"°F")) for i in range(net_cooling_data.number_of_stages)],
    net_cooling_cop_rated = [net_cooling_data.get_cop[i](fr_u(95.0,"°F")) for i in range(net_cooling_data.number_of_stages)],
    gross_cooling_cop_rated  =  None, # Use net instead
    fan_efficacy_cooling_rated = fan_efficacy_cooling_rated,
    flow_rated_per_cap_cooling_rated = flow_rated_per_cap_cooling_rated,
    c_d_cooling = c_d_cooling,
    net_heating_capacity_rated = [net_heating_data.get_capacity[i](fr_u(47.0,"°F")) for i in range(net_heating_data.number_of_stages)],
    net_heating_cop_rated = [net_heating_data.get_cop[i](fr_u(47.0,"°F")) for i in range(net_heating_data.number_of_stages)],
    gross_heating_cop_rated  =  None, # Use net instead
    fan_efficacy_heating_rated = fan_efficacy_heating_rated,
    flow_rated_per_cap_heating_rated = flow_rated_per_cap_heating_rated,
    c_d_heating = c_d_heating,
    full_load_speed = 1 if net_cooling_data.number_of_stages > 3 else 0,
    intermediate_speed = net_cooling_data.number_of_stages - 2
    )
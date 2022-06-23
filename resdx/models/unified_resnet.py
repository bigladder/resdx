from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .henderson_defrost_model import HendersonDefrostModel
from ..units import fr_u
from ..fan import ConstantEfficacyFan, ECMFlowFan, PSCFan

class RESNETDXModel(DXModel):

  # Power and capacity
  def gross_cooling_power(self, conditions):
    return NRELDXModel.gross_cooling_power(self, conditions)

  def gross_total_cooling_capacity(self, conditions):
    return NRELDXModel.gross_total_cooling_capacity(self, conditions)

  def gross_sensible_cooling_capacity(self, conditions):
    return NRELDXModel.gross_sensible_cooling_capacity(self, conditions)

  def gross_shr(self, conditions):
    return Title24DXModel.gross_shr(self, conditions)

  def gross_steady_state_heating_capacity(self, conditions):
    return NRELDXModel.gross_steady_state_heating_capacity(self, conditions)

  def gross_integrated_heating_capacity(self, conditions):
    return HendersonDefrostModel.gross_integrated_heating_capacity(self, conditions)

  def gross_steady_state_heating_power(self, conditions):
    return NRELDXModel.gross_steady_state_heating_power(self, conditions)

  def gross_integrated_heating_power(self, conditions):
    return HendersonDefrostModel.gross_integrated_heating_power(self, conditions)

  def gross_cooling_power_charge_factor(self, conditions):
    return NRELDXModel.gross_cooling_power_charge_factor(self, conditions)

  def gross_total_cooling_capacity_charge_factor(self, conditions):
    return NRELDXModel.gross_total_cooling_capacity_charge_factor(self, conditions)

  def gross_steady_state_heating_capacity_charge_factor(self, conditions):
    return NRELDXModel.gross_steady_state_heating_capacity_charge_factor(self, conditions)

  def gross_steady_state_heating_power_charge_factor(self, conditions):
    return NRELDXModel.gross_steady_state_heating_power_charge_factor(self, conditions)

  def set_net_capacities_and_fan(self, rated_net_total_cooling_capacity, rated_net_heating_capacity, fan):
    # Set high speed capacities

    ## Cooling (high speed net total cooling capacity is required)
    if type(rated_net_total_cooling_capacity) is list:
      self.system.rated_net_total_cooling_capacity = rated_net_total_cooling_capacity
    else:
      # Even if the system has more than one speed, the first speed will be the input
      self.system.rated_net_total_cooling_capacity = [rated_net_total_cooling_capacity]

    ## Heating
    rated_net_heating_capacity = self.set_default(rated_net_heating_capacity, self.system.rated_net_total_cooling_capacity[0]*0.98 + fr_u(180.,"Btu/hr")) # From Title24
    if type(rated_net_heating_capacity) is list:
      self.system.rated_net_heating_capacity = rated_net_heating_capacity
    else:
      # Even if the system has more than one speed, the first speed will be the input
      self.system.rated_net_heating_capacity = [rated_net_heating_capacity]

    # setup fan
    self.system.rated_full_flow_external_static_pressure = self.system.get_rated_full_flow_rated_pressure()

    # Rated flow rates per net capacity
    self.system.rated_cooling_airflow_per_rated_net_capacity = self.set_default(self.system.rated_cooling_airflow_per_rated_net_capacity, [fr_u(400.,"cfm/ton_ref")]*self.system.number_of_input_stages)
    self.system.rated_heating_airflow_per_rated_net_capacity = self.set_default(self.system.rated_heating_airflow_per_rated_net_capacity, [fr_u(400.,"cfm/ton_ref")]*self.system.number_of_input_stages)

    if fan is not None:
      self.system.fan = fan
      for i in range(self.system.number_of_input_stages):
        self.system.rated_cooling_airflow[i] = self.system.fan.airflow(self.system.rated_cooling_fan_speed[i])
        self.system.rated_cooling_fan_power[i] = self.system.fan.power(self.system.rated_cooling_fan_speed[i])
      for i in range(self.system.number_of_input_stages):
        self.system.rated_heating_airflow[i] = self.system.fan.airflow(self.system.rated_heating_fan_speed[i])
        self.system.rated_heating_fan_power[i] = self.system.fan.power(self.system.rated_heating_fan_speed[i])
    else:
      self.system.cooling_fan_speed = [None]*self.system.number_of_input_stages
      self.system.heating_fan_speed = [None]*self.system.number_of_input_stages
      self.system.rated_cooling_fan_speed = [None]*self.system.number_of_input_stages
      self.system.rated_heating_fan_speed = [None]*self.system.number_of_input_stages

      self.system.rated_cooling_airflow[0] = self.system.rated_net_total_cooling_capacity[0]*self.system.rated_cooling_airflow_per_rated_net_capacity[0]
      design_external_static_pressure = fr_u(0.5, "in_H2O")
      if self.system.number_of_input_stages > 1:
        self.system.fan = ECMFlowFan(self.system.rated_cooling_airflow[0], design_external_static_pressure, design_efficacy=fr_u(0.3, 'W/cfm'))
      else:
        self.system.fan = PSCFan(self.system.rated_cooling_airflow[0], design_external_static_pressure, design_efficacy=fr_u(0.365, 'W/cfm'))

      self.system.cooling_fan_speed[0] = self.system.fan.number_of_speeds - 1

      self.system.rated_heating_airflow[0] = self.system.rated_net_heating_capacity[0]*self.system.rated_heating_airflow_per_rated_net_capacity[0]
      self.system.fan.add_speed(self.system.rated_heating_airflow[0])
      self.system.heating_fan_speed[0] = self.system.fan.number_of_speeds - 1

      # At rated pressure
      self.system.rated_cooling_external_static_pressure[0] = self.system.calculate_rated_pressure(self.system.rated_cooling_airflow[0])
      self.system.fan.add_speed(self.system.rated_cooling_airflow[0], external_static_pressure=self.system.rated_cooling_external_static_pressure[0])
      self.system.rated_cooling_fan_speed[0] = self.system.fan.number_of_speeds - 1
      self.system.rated_cooling_fan_power[0] = self.system.fan.power(self.system.rated_cooling_fan_speed[0],self.system.rated_cooling_external_static_pressure[0])

      self.system.rated_heating_external_static_pressure[0] = self.system.calculate_rated_pressure(self.system.rated_heating_airflow[0])
      self.system.fan.add_speed(self.system.rated_heating_airflow[0], external_static_pressure=self.system.rated_heating_external_static_pressure[0])
      self.system.rated_heating_fan_speed[0] = self.system.fan.number_of_speeds - 1
      self.system.rated_heating_fan_power[0] = self.system.fan.power(self.system.rated_heating_fan_speed[0],self.system.rated_heating_external_static_pressure[0])


      # if net cooling capacities are provided for other speeds, add corresponding fan speeds
      for i, net_capacity in enumerate(self.system.rated_net_total_cooling_capacity[1:]):
        i += 1 # Since we're starting at the second item
        self.system.rated_cooling_airflow[i] = net_capacity*self.system.rated_cooling_airflow_per_rated_net_capacity[i]
        self.system.fan.add_speed(self.system.rated_cooling_airflow[i])
        self.system.cooling_fan_speed[i] = self.system.fan.number_of_speeds - 1

        # At rated pressure
        self.system.rated_cooling_external_static_pressure[i] = self.system.calculate_rated_pressure(self.system.rated_cooling_airflow[i])
        self.system.fan.add_speed(self.system.rated_cooling_airflow[i], external_static_pressure=self.system.rated_cooling_external_static_pressure[i])
        self.system.rated_cooling_fan_speed[i] = self.system.fan.number_of_speeds - 1
        self.system.rated_cooling_fan_power[i] = self.system.fan.power(self.system.rated_cooling_fan_speed[i],self.system.rated_cooling_external_static_pressure[i])

      # if net cooling capacities are provided for other speeds, add corresponding fan speeds
      for i, net_capacity in enumerate(self.system.rated_net_heating_capacity[1:]):
        i += 1 # Since we're starting at the second item
        self.system.rated_heating_airflow[i] = self.system.rated_net_heating_capacity[i]*self.system.rated_heating_airflow_per_rated_net_capacity[i]
        self.system.fan.add_speed(self.system.rated_heating_airflow[i])
        self.system.heating_fan_speed[i] = self.system.fan.number_of_speeds - 1

        # At rated pressure
        self.system.rated_heating_external_static_pressure[i] = self.system.calculate_rated_pressure(self.system.rated_heating_airflow[i])
        self.system.fan.add_speed(self.system.rated_heating_airflow[i], external_static_pressure=self.system.rated_heating_external_static_pressure[i])
        self.system.rated_heating_fan_speed[i] = self.system.fan.number_of_speeds - 1
        self.system.rated_heating_fan_power[i] = self.system.fan.power(self.system.rated_heating_fan_speed[i],self.system.rated_heating_external_static_pressure[i])


    # setup lower speed net capacities if they aren't provided
    if len(self.system.rated_net_total_cooling_capacity) < self.system.number_of_input_stages:
      if self.system.number_of_input_stages == 2:
        # Cooling
        cooling_capacity_ratio = 0.72
        self.system.rated_cooling_fan_power[0] = self.system.fan.power(self.system.rated_cooling_fan_speed[0],self.system.rated_cooling_external_static_pressure[0])
        self.system.rated_gross_total_cooling_capacity[0] = self.system.rated_net_total_cooling_capacity[0] + self.system.rated_cooling_fan_power[0]
        self.system.rated_gross_total_cooling_capacity[1] = self.system.rated_gross_total_cooling_capacity[0]*cooling_capacity_ratio

        # Solve for rated flow rate
        guess_airflow = self.system.fan.design_airflow[self.system.cooling_fan_speed[0]]*cooling_capacity_ratio
        self.system.rated_cooling_airflow[1] = self.system.fan.find_rated_fan_speed(self.system.rated_gross_total_cooling_capacity[1], self.system.rated_heating_airflow_per_rated_net_capacity[1], guess_airflow, self.system.rated_full_flow_external_static_pressure)
        self.system.rated_cooling_fan_speed[1] = self.system.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.system.fan.add_speed(self.system.rated_cooling_airflow[1])
        self.system.cooling_fan_speed[1] = self.system.fan.number_of_speeds - 1

        self.system.rated_cooling_external_static_pressure[1] = self.system.calculate_rated_pressure(self.system.rated_cooling_airflow[1])
        self.system.rated_cooling_fan_power[1] = self.system.fan.power(self.system.rated_cooling_fan_speed[1],self.system.rated_cooling_external_static_pressure[1])
        self.system.rated_net_total_cooling_capacity.append(self.system.rated_gross_total_cooling_capacity[1] - self.system.rated_cooling_fan_power[1])

        # Heating
        heating_capacity_ratio = 0.72
        self.system.rated_heating_fan_power[0] = self.system.fan.power(self.system.rated_heating_fan_speed[0],self.system.rated_heating_external_static_pressure[0])
        self.system.rated_gross_heating_capacity[0] = self.system.rated_net_heating_capacity[0] - self.system.rated_heating_fan_power[0]
        self.system.rated_gross_heating_capacity[1] = self.system.rated_gross_heating_capacity[0]*heating_capacity_ratio

        # Solve for rated flow rate
        guess_airflow = self.system.fan.design_airflow[self.system.heating_fan_speed[0]]*heating_capacity_ratio
        self.system.rated_heating_airflow[1] = self.system.fan.find_rated_fan_speed(self.system.rated_gross_heating_capacity[1], self.system.rated_heating_airflow_per_rated_net_capacity[1], guess_airflow, self.system.rated_full_flow_external_static_pressure)
        self.system.rated_heating_fan_speed[1] = self.system.fan.number_of_speeds - 1

        # Add fan setting for design pressure
        self.system.fan.add_speed(self.system.rated_heating_airflow[1])
        self.system.heating_fan_speed[1] = self.system.fan.number_of_speeds - 1

        self.system.rated_heating_external_static_pressure[1] = self.system.calculate_rated_pressure(self.system.rated_heating_airflow[1])
        self.system.rated_heating_fan_power[1] = self.system.fan.power(self.system.rated_heating_fan_speed[1],self.system.rated_heating_external_static_pressure[1])
        self.system.rated_net_heating_capacity.append(self.system.rated_gross_heating_capacity[1] + self.system.rated_heating_fan_power[1])
      else:
        raise Exception(f"No default rated net total cooling capacities for systems with more than two speeds")

  def set_c_d_cooling(self, input):
    if self.system.input_seer is None:
      default = 0.25
    else:
      default = RESNETDXModel.c_d(self.system.input_seer)
    self.system.c_d_cooling = self.set_default(input, default)

  def set_c_d_heating(self, input):
    if self.system.input_hspf is None:
      default = 0.25
    else:
      default = RESNETDXModel.c_d(RESNETDXModel.estimated_seer(self.system.input_hspf))
    self.system.c_d_heating = self.set_default(input, default)

  def set_rated_net_cooling_cop(self, input):
    NRELDXModel.set_rated_net_cooling_cop(self, input)

  def set_rated_gross_cooling_cop(self, input):
    NRELDXModel.set_rated_gross_cooling_cop(self, input)

  def set_rated_net_heating_cop(self, input):
    NRELDXModel.set_rated_net_heating_cop(self, input)

  def set_rated_gross_heating_cop(self, input):
    NRELDXModel.set_rated_gross_heating_cop(self, input)

  @staticmethod
  def fan_efficacy(seer):
      if seer <= 14:
          return fr_u(0.25,'W/cfm')
      elif seer >= 16:
          return fr_u(0.18,'W/cfm')
      else:
          return fr_u(0.25,'W/cfm') + (fr_u(0.18,'W/cfm') - fr_u(0.25,'W/cfm'))/2.0 * (seer - 14.0)

  @staticmethod
  def c_d(seer):
      if seer <= 12:
          return 0.2
      elif seer >= 13:
          return 0.1
      else:
          return 0.2 + (0.1 - 0.2)*(seer - 12.0)

  @staticmethod
  def estimated_seer(hspf): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
      return (hspf - 3.2627)/0.3526

  @staticmethod
  def estimated_hspf(seer): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
      return seer*0.3526 + 3.2627


from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .defrost_model import DEFROSTModel

class RESNETDXModel(DXModel):
  @staticmethod
  def gross_cooling_power(conditions, system):
    return NRELDXModel.gross_cooling_power(conditions, system)

  @staticmethod
  def gross_total_cooling_capacity(conditions, system):
    return NRELDXModel.gross_total_cooling_capacity(conditions, system)

  @staticmethod
  def gross_sensible_cooling_capacity(conditions, system):
    return NRELDXModel.gross_sensible_cooling_capacity(conditions, system)

  @staticmethod
  def gross_shr(conditions):
    return Title24DXModel.gross_shr(conditions)

  @staticmethod
  def gross_steady_state_heating_capacity(conditions, system):
    return NRELDXModel.gross_steady_state_heating_capacity(conditions, system)

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    return DEFROSTModel.gross_integrated_heating_capacity(conditions, system)

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    return NRELDXModel.gross_steady_state_heating_power(conditions, system)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    return DEFROSTModel.gross_integrated_heating_power(conditions, system)
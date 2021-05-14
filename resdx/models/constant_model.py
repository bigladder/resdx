from scipy import optimize

from ..units import fr_u, to_u
from ..util import calc_biquad, calc_quad
from ..psychrometrics import psychrolib, PsychState
from ..defrost import DefrostControl, DefrostStrategy
from ..conditions import CoolingConditions

from .base_model import DXModel

class ConstantDXModel(DXModel):

  '''This model is developed for testing purposes where the performance is constant across all conditions'''

  @staticmethod
  def gross_cooling_power(conditions, system):
    return system.gross_total_cooling_capacity_rated[conditions.compressor_speed]/system.gross_cooling_cop_rated[conditions.compressor_speed]

  @staticmethod
  def gross_total_cooling_capacity(conditions, system):
    return system.gross_total_cooling_capacity_rated[conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    return system.gross_heating_capacity_rated[conditions.compressor_speed]/system.gross_heating_cop_rated[conditions.compressor_speed]

  @staticmethod
  def gross_steady_state_heating_capacity(conditions, system):
    return system.gross_heating_capacity_rated[conditions.compressor_speed]

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    return system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def gross_sensible_cooling_capacity(conditions, system):
    return system.gross_total_cooling_capacity(conditions)*gross_shr(conditions)

  @staticmethod
  def gross_shr(conditions):
    return 0.7



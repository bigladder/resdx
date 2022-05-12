from .base_model import DXModel
from .nrel import NRELDXModel
from .title24 import Title24DXModel
from .henderson_defrost_model import HendersonDefrostModel
from ..units import fr_u, to_u

def fan_efficacy(seer):
    if seer <= 14:
        return fr_u(0.25,'W/(cu_ft/min)')
    elif seer >= 16:
        return fr_u(0.18,'W/(cu_ft/min)')
    else:
        return fr_u(0.25,'W/(cu_ft/min)') + (fr_u(0.18,'W/(cu_ft/min)') - fr_u(0.25,'W/(cu_ft/min)'))/2.0 * (seer - 14.0)


def c_d(seer):
    if seer <= 12:
        return 0.2
    elif seer >= 13:
        return 0.1
    else:
        return 0.2 + (0.1 - 0.2)*(seer - 12.0)

def estimated_seer(hspf): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
    return (hspf - 3.2627)/0.3526


class RESNETDXModel(DXModel):

  # Power and capacity
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
    return HendersonDefrostModel.gross_integrated_heating_capacity(conditions, system)

  @staticmethod
  def gross_steady_state_heating_power(conditions, system):
    return NRELDXModel.gross_steady_state_heating_power(conditions, system)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    return HendersonDefrostModel.gross_integrated_heating_power(conditions, system)
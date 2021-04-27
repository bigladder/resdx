import sys

from scipy import optimize
import numpy as np
from scipy import interpolate

from ..units import fr_u, to_u
from ..util import calc_biquad, calc_quad
from ..psychrometrics import psychrolib, PsychState
from ..defrost import DefrostControl, DefrostStrategy
from ..conditions import CoolingConditions

from .base_model import DXModel

defrost_temp_array = [fr_u(-20,"°F"), fr_u(17,"°F"), fr_u(27,"°F"), fr_u(37,"°F")]

class HendersonDefrostModel(DXModel):

  '''Based on Piotr A. Domanski et al: Sensitivity Analysis of Installation Faults on Heat Pump Performance (NIST Technical Note 1848)
     and Hugh I. Henderson et al: Savings Calculations for Residential Air Source Heat Pumps (NYSERDA and NYS Department of Public Service)'''

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):

    if system.defrost.in_defrost(conditions):
      return system.gross_steady_state_heating_capacity(conditions) * (1-HendersonDefrostModel.fdef(conditions,system.defrost.high_temperature))
    else:
      return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    #
      return system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def fdef(conditions, max_defrost_outdoor_temperature):
      epsilon = 0.0000000000001
      fdef_values = [0.075, 0.085, 0.11, 0.09]
      temp_array = defrost_temp_array + [max_defrost_outdoor_temperature]
      temp_array.sort()
      max_defrost_outdoor_temperature_index = temp_array.index(max_defrost_outdoor_temperature)

      f = interpolate.interp1d(defrost_temp_array , [0.075, 0.085, 0.11, 0.09],bounds_error = False, fill_value=(0.075,0.09))
      fdef_max_defrost_outdoor_temperature = f(max_defrost_outdoor_temperature)

      x = temp_array[0:max_defrost_outdoor_temperature_index+1] + [max_defrost_outdoor_temperature+epsilon]
      y = fdef_values[0:max_defrost_outdoor_temperature_index] + [fdef_max_defrost_outdoor_temperature] + [0]
      g = interpolate.interp1d(x, y,bounds_error = False, fill_value=(0.075,0))
      return g(conditions.outdoor.db)
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

class HendersonDefrostModel(DXModel):

  '''Based on Piotr A. Domanski et al: Sensitivity Analysis of Installation Faults on Heat Pump Performance (NIST Technical Note 1848)
     and Hugh I. Henderson et al: Savings Calculations for Residential Air Source Heat Pumps (NYSERDA and NYS Department of Public Service)'''


  defrost_temperatures = [fr_u(-20,"°F"), fr_u(17,"°F"), fr_u(27,"°F"), fr_u(37,"°F")]
  defrost_fractions = [0.075, 0.085, 0.11, 0.09]

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):

    if system.defrost.in_defrost(conditions):
      return system.gross_steady_state_heating_capacity(conditions) * (1-HendersonDefrostModel.fdef(conditions))
    else:
      return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
      return system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def fdef(conditions):
      return interpolate.interp1d(
        HendersonDefrostModel.defrost_temperatures,
        HendersonDefrostModel.defrost_fractions,
        bounds_error = False,
        fill_value=(HendersonDefrostModel.defrost_fractions[0],HendersonDefrostModel.defrost_fractions[-1])
        )(conditions.outdoor.db)
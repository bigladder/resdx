from scipy import optimize
import numpy as np

from ..units import fr_u, to_u
from ..util import calc_biquad, calc_quad
from ..psychrometrics import psychrolib, PsychState
from ..defrost import DefrostControl, DefrostStrategy
from ..conditions import CoolingConditions

from .base_model import DXModel

class DEFROSTModel(DXModel):

  '''Based on Piotr A. Domanski et al: Sensitivity Analysis of Installation Faults on Heat Pump Performance (NIST Technical Note 1848)
     and Hugh I. Henderson et al: Savings Calculations for Residential Air Source Heat Pumps (NYSERDA and NYS Department of Public Service)'''

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):

    if system.defrost.in_defrost(conditions):
      return system.gross_steady_state_heating_capacity(conditions) * (1-DEFROSTModel.fdef(conditions,to_u(system.defrost.high_temperature,"°F")))
    else:
      return system.gross_steady_state_heating_capacity(conditions)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    #
      return system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def fdef(conditions, cut_off_temp): # cut_off_temp in F
      Todb = to_u(conditions.outdoor.db,"°F")
      if Todb <= -20:
          return 0.075
      elif Todb > -20 and Todb <= 17:
          return np.interp(Todb, [-20, 17], [0.075, 0.085])
      elif Todb > 17 and Todb <= 27:
          return np.interp(Todb, [17, 27], [0.085, 0.11])
      elif Todb > 27 and Todb <= 37:
          return np.interp(Todb, [27, 37], [0.11, 0.09])
      elif Todb > 37 and Todb < cut_off_temp: #Todb <= 47:
          return 0.09
      elif Todb > cut_off_temp:
          return 0.0
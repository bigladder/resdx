import sys

from scipy import interpolate

from koozie import fr_u

from .base_model import DXModel

class HendersonDefrostModel(DXModel):

  '''Based on Piotr A. Domanski et al: Sensitivity Analysis of Installation Faults on Heat Pump Performance (NIST Technical Note 1848)
     and Hugh I. Henderson et al: Savings Calculations for Residential Air Source Heat Pumps (NYSERDA and NYS Department of Public Service)'''


  defrost_temperatures = [fr_u(-20,"°F"), fr_u(17,"°F"), fr_u(27,"°F"), fr_u(37,"°F")]
  defrost_fractions = [0.075, 0.085, 0.11, 0.09]

  def gross_integrated_heating_capacity(self, conditions):

    if self.system.defrost.in_defrost(conditions):
      return self.system.gross_steady_state_heating_capacity(conditions) * (1-HendersonDefrostModel.fdef(conditions))
    else:
      return self.system.gross_steady_state_heating_capacity(conditions)

  def gross_integrated_heating_power(self, conditions):
      return self.system.gross_steady_state_heating_power(conditions)

  @staticmethod
  def fdef(conditions):
      return interpolate.interp1d(
        HendersonDefrostModel.defrost_temperatures,
        HendersonDefrostModel.defrost_fractions,
        bounds_error = False,
        fill_value=(HendersonDefrostModel.defrost_fractions[0],HendersonDefrostModel.defrost_fractions[-1])
        )(conditions.outdoor.db)

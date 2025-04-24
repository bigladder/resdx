from scipy import interpolate

from koozie import fr_u

from ..dx_unit import DXUnit


class CarrierDefrostModel(DXUnit):
    """Based on Piotr A. Domanski et al: Sensitivity Analysis of Installation Faults on Heat Pump Performance (NIST Technical Note 1848)
    and Hugh I. Henderson et al: Savings Calculations for Residential Air Source Heat Pumps (NYSERDA and NYS Department of Public Service)
    """

    defrost_temperatures = [
        fr_u(-20, "°F"),
        fr_u(17, "°F"),
        fr_u(27, "°F"),
        fr_u(37, "°F"),
    ]
    defrost_fractions = [0.075, 0.085, 0.11, 0.09]

    def full_charge_gross_integrated_heating_capacity(self, conditions):
        if self.defrost.in_defrost(conditions):
            return self.full_charge_gross_steady_state_heating_capacity(conditions) * (
                1 - CarrierDefrostModel.fdef(conditions)
            )
        else:
            return self.full_charge_gross_steady_state_heating_capacity(conditions)

    def full_charge_gross_integrated_heating_power(self, conditions):
        return self.full_charge_gross_steady_state_heating_power(conditions)

    @staticmethod
    def fdef(conditions):
        return interpolate.interp1d(
            CarrierDefrostModel.defrost_temperatures,
            CarrierDefrostModel.defrost_fractions,
            bounds_error=False,
            fill_value=(
                CarrierDefrostModel.defrost_fractions[0],
                CarrierDefrostModel.defrost_fractions[-1],
            ),
        )(conditions.outdoor.db)

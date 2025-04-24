from ..dx_unit import DXUnit


class ConstantDXModel(DXUnit):
    """This model is developed for testing purposes where the performance is constant across all conditions"""

    def full_charge_gross_cooling_power(self, conditions):
        return (
            self.rated_gross_total_cooling_capacity[conditions.compressor_speed]
            / self.rated_gross_cooling_cop[conditions.compressor_speed]
        )

    def full_charge_gross_total_cooling_capacity(self, conditions):
        return self.rated_gross_total_cooling_capacity[conditions.compressor_speed]

    def full_charge_gross_steady_state_heating_power(self, conditions):
        return (
            self.rated_gross_heating_capacity[conditions.compressor_speed]
            / self.rated_gross_heating_cop[conditions.compressor_speed]
        )

    def full_charge_gross_steady_state_heating_capacity(self, conditions):
        return self.rated_gross_heating_capacity[conditions.compressor_speed]

    def full_charge_gross_integrated_heating_capacity(self, conditions):
        return self.gross_steady_state_heating_capacity(conditions)

    def full_charge_gross_integrated_heating_power(self, conditions):
        return self.gross_steady_state_heating_power(conditions)

    def full_charge_gross_sensible_cooling_capacity(self, conditions):
        return self.gross_total_cooling_capacity(conditions) * self.gross_shr(conditions)

    def gross_shr(self, conditions):
        return 0.7

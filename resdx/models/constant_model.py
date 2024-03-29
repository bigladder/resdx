from .base_model import DXModel


class ConstantDXModel(DXModel):

    """This model is developed for testing purposes where the performance is constant across all conditions"""

    def gross_cooling_power(self, conditions):
        return (
            self.system.rated_gross_total_cooling_capacity[conditions.compressor_speed]
            / self.system.rated_gross_cooling_cop[conditions.compressor_speed]
        )

    def gross_total_cooling_capacity(self, conditions):
        return self.system.rated_gross_total_cooling_capacity[
            conditions.compressor_speed
        ]

    def gross_steady_state_heating_power(self, conditions):
        return (
            self.system.rated_gross_heating_capacity[conditions.compressor_speed]
            / self.system.rated_gross_heating_cop[conditions.compressor_speed]
        )

    def gross_steady_state_heating_capacity(self, conditions):
        return self.system.rated_gross_heating_capacity[conditions.compressor_speed]

    def gross_integrated_heating_capacity(self, conditions):
        return self.system.gross_steady_state_heating_capacity(conditions)

    def gross_integrated_heating_power(self, conditions):
        return self.system.gross_steady_state_heating_power(conditions)

    def gross_sensible_cooling_capacity(self, conditions):
        return self.system.gross_total_cooling_capacity(conditions) * self.gross_shr(
            conditions
        )

    def gross_shr(self, conditions):
        return 0.7

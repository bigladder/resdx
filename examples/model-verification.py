# %%
import resdx

# This example compares just the defrost characteristics (everything else is held constant)


class DefrostVerificationModelNREL(
    resdx.models.ConstantDXModel
):  # Implemented in EnergyPlus
    def gross_integrated_heating_capacity(self, conditions):
        return resdx.models.NRELDXModel.gross_integrated_heating_capacity(
            self, conditions
        )

    def gross_integrated_heating_power(self, conditions):
        return resdx.models.NRELDXModel.gross_integrated_heating_power(self, conditions)


test_dx_system = resdx.DXUnit(
    model=DefrostVerificationModelNREL(),  # These inputs should be similar to EnergyPlus model.
    rated_net_heating_cop=[5],
    rated_net_heating_capacity=[10600],
)


class DefrostVerificationModelCarrier(resdx.models.ConstantDXModel):
    def gross_integrated_heating_capacity(self, conditions):
        return resdx.models.CarrierDefrostModel.gross_integrated_heating_capacity(
            self, conditions
        )

    def gross_integrated_heating_power(self, conditions):
        return resdx.models.CarrierDefrostModel.gross_integrated_heating_power(
            self, conditions
        )


test_dx_system_carrier = resdx.DXUnit(
    model=DefrostVerificationModelCarrier(),
    rated_net_heating_cop=[5],
    rated_net_heating_capacity=[10600],
)


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import numpy as np

# Plot integrated power and capacity
T_out = np.arange(-9.4, 71.6, (71.6 + 9.4) / 365)
conditions = [
    test_dx_system.make_condition(
        resdx.HeatingConditions,
        outdoor=resdx.PsychState(drybulb=resdx.fr_u(T, "°F"), rel_hum=0.4),
    )
    for T in T_out
]  # Same artificial weather used in EnergyPlus.
Q_nrel = [
    test_dx_system.gross_integrated_heating_capacity(condition)
    for condition in conditions
]
#
Carrier = [
    test_dx_system_carrier.gross_integrated_heating_capacity(condition)
    for condition in conditions
]
#
# To create a standalone figure window

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Temp (°F)")
ax1.set_ylabel("Integrated Capacity (W)", color=color)
ax1.plot(T_out, Q_nrel, color=color)
ax1.plot(T_out, Carrier, color="b")
ax1.legend(["NREL", "Carrier"])
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim([0, 11000])

fig.tight_layout()
plt.savefig("output/verification-model-heat-pump.png")

# %%

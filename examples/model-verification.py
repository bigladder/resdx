# %%
import resdx

# This example compares just the defrost characteristics (everything else is held constant)


class DefrostVerificationModel_nrel(
    resdx.models.ConstantDXModel
):  # Implemented in EnergyPlus
    def gross_integrated_heating_capacity(self, conditions):
        return resdx.models.NRELDXModel.gross_integrated_heating_capacity(
            self, conditions
        )

    def gross_integrated_heating_power(self, conditions):
        return resdx.models.NRELDXModel.gross_integrated_heating_power(self, conditions)


test_dx_system = resdx.DXUnit(
    model=DefrostVerificationModel_nrel(),  # These inputs should be similar to EnergyPlus model.
    rated_net_heating_cop=[5],
    rated_net_heating_capacity=[10600],
)


class DefrostVerificationModel_henderson(resdx.models.ConstantDXModel):
    def gross_integrated_heating_capacity(self, conditions):
        return resdx.models.HendersonDefrostModel.gross_integrated_heating_capacity(
            self, conditions
        )

    def gross_integrated_heating_power(self, conditions):
        return resdx.models.HendersonDefrostModel.gross_integrated_heating_power(
            self, conditions
        )


test_dx_system_henderson = resdx.DXUnit(
    model=DefrostVerificationModel_henderson(),
    rated_net_heating_cop=[5],
    rated_net_heating_capacity=[10600],
)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import numpy as np

# Plot integrated power and capacity
T_out = np.arange(-23, 22, (23 + 22) / 365)
conditions = [
    test_dx_system.make_condition(
        resdx.HeatingConditions,
        outdoor=resdx.PsychState(drybulb=resdx.fr_u(T, "°C"), rel_hum=0.4),
    )
    for T in T_out
]  # Same artificial weather used in EnergyPlus.
Q_nrel = [
    test_dx_system.gross_integrated_heating_capacity(condition)
    for condition in conditions
]
#
Qhenderson = [
    test_dx_system_henderson.gross_integrated_heating_capacity(condition)
    for condition in conditions
]
#
# To create a standalone figure window

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Temp (°C)")
ax1.set_ylabel("Integrated Capacity (W)", color=color)
ax1.plot(T_out, Q_nrel, color=color)
ax1.plot(T_out, Qhenderson, color="b")
ax1.legend(["NREL", "Henderson"])
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim([7000, 11000])

fig.tight_layout()
plt.savefig("output/verification-model-heat-pump.png")

# %%

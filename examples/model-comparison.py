# %%
import resdx

seer = 14.0
hspf = 7.7
# cooling factor:
# heating factor:

t24_unit = resdx.DXUnit(
    model=resdx.models.Title24DXModel(),
    rated_net_cooling_cop=[
        resdx.fr_u(resdx.models.Title24DXModel.eer_rated(seer), "Btu/Wh")
    ],
    c_d_cooling=0.0,
    input_seer=seer,
    rated_net_heating_cop=[resdx.models.Title24DXModel.cop47_rated(hspf)],
    c_d_heating=resdx.models.Title24DXModel.c_d_heating(hspf),
    input_hspf=hspf,
    # cap17=[resdx.fr_u(2.0,'ton_ref')],
    defrost=resdx.Defrost(
        high_temperature=resdx.fr_u(45.0, "°F"), low_temperature=resdx.fr_u(17.0, "°F")
    ),
)

t24_unit.print_cooling_info()
t24_unit.print_heating_info()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import numpy as np

# Plot integrated power and capacity
T_out = np.arange(-23, 76, 1)
conditions = [
    t24_unit.make_condition(
        resdx.HeatingConditions,
        outdoor=resdx.PsychState(drybulb=resdx.fr_u(T, "°F"), rel_hum=0.4),
    )
    for T in T_out
]
Q_integrated = [
    t24_unit.gross_integrated_heating_capacity(condition) for condition in conditions
]
P_integrated = [
    t24_unit.gross_integrated_heating_power(condition) for condition in conditions
]
COP_integrated = [
    t24_unit.gross_integrated_heating_cop(condition) for condition in conditions
]

fig, ax1 = plt.subplots()

color = "tab:red"
ax1.set_xlabel("Temp (°F)")
ax1.set_ylabel("Capacity/Power (W)", color=color)
ax1.plot(T_out, Q_integrated, color=color)
ax1.plot(T_out, P_integrated, color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim([0, 15000])

ax2 = ax1.twinx()

color = "tab:blue"
ax2.set_ylabel("COP", color=color)
ax2.plot(T_out, COP_integrated, color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim([0, 5.5])

fig.tight_layout()
plt.savefig("output/title24-heat-pump.png")

# %%

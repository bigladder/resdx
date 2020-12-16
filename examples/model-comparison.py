#%%
import resdx

seer = 14.0
hspf = 7.7
# cooling factor:
# heating factor:

t24_unit = resdx.DXUnit(gross_total_cooling_capacity_fn=resdx.title24_gross_total_cooling_capacity,
                        gross_sensible_cooling_capacity_fn=resdx.title24_gross_sensible_cooling_capacity,
                        gross_cooling_power_fn=resdx.title24_gross_cooling_power,
                        net_cooling_cop_rated=[resdx.u(resdx.title24_eer_rated(seer),'Btu/Wh')],
                        fan_eff_cooling_rated=[resdx.u(0.365,'W/(cu_ft/min)')],
                        c_d_cooling=0.0,
                        input_seer=seer,
                        net_heating_cop_rated=[resdx.title24_cop47_rated(hspf)],
                        gross_steady_state_heating_capacity_fn=resdx.title24_gross_steady_state_heating_capacity,
                        gross_steady_state_heating_power_fn=resdx.title24_gross_steady_state_heating_power,
                        gross_integrated_heating_capacity_fn=resdx.title24_gross_integrated_heating_capacity,
                        gross_integrated_heating_power_fn=resdx.title24_gross_integrated_heating_power,
                        input_hspf=hspf
)

t24_unit.seer()
t24_unit.hspf()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np

# Plot integrated power and capacity
T_out = np.arange(-23,75+1,1)
conditions = [t24_unit.make_condition(resdx.HeatingConditions,outdoor=resdx.PsychState(drybulb=resdx.u(T,"°F"),wetbulb=resdx.u(T-2.0,"°F"))) for T in T_out]
Q_integrated = [t24_unit.gross_integrated_heating_capacity(condition) for condition in conditions]
P_integrated = [t24_unit.gross_integrated_heating_power(condition) for condition in conditions]
COP_integrated = [t24_unit.gross_integrated_heating_cop(condition) for condition in conditions]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Temp (°F)')
ax1.set_ylabel('Capacity/Power (W)', color=color)
ax1.plot(T_out, Q_integrated, color=color)
ax1.plot(T_out, P_integrated, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('COP', color=color)
ax2.plot(T_out, COP_integrated, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig('output/title24-heat-pump.png')

# %%

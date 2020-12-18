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
                        fan_eff_cooling_rated=[resdx.u(0.58,'W/(cu_ft/min)')],
                        flow_rated_per_cap_cooling_rated=[resdx.u(350.0,"(cu_ft/min)/ton_of_refrigeration")],
                        c_d_cooling=0.0,
                        input_seer=seer,
                        net_heating_cop_rated=[resdx.title24_cop47_rated(hspf)],
                        gross_steady_state_heating_capacity_fn=resdx.title24_gross_steady_state_heating_capacity,
                        gross_steady_state_heating_power_fn=resdx.title24_gross_steady_state_heating_power,
                        gross_integrated_heating_capacity_fn=resdx.title24_gross_integrated_heating_capacity,
                        gross_integrated_heating_power_fn=resdx.title24_gross_integrated_heating_power,
                        flow_rated_per_cap_heating_rated=[resdx.u(350.0,"(cu_ft/min)/ton_of_refrigeration")],
                        fan_eff_heating_rated=[resdx.u(0.58,'W/(cu_ft/min)')],
                        c_d_heating=resdx.title24_c_d_heating(hspf),
                        input_hspf=hspf,
                        #cap17=[resdx.u(2.0,'ton_of_refrigeration')],
                        defrost=resdx.Defrost(high_temperature=resdx.u(45.0,"°F"), low_temperature=resdx.u(17.0,"°F"))
)

t24_unit.print_cooling_info()
t24_unit.print_heating_info()
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
ax1.set_ylim([0,15000])

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('COP', color=color)
ax2.plot(T_out, COP_integrated, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0,5.5])

fig.tight_layout()
plt.savefig('output/title24-heat-pump.png')

# %%

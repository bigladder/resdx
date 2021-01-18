#%%
import resdx

VerificationModel = resdx.DXUnit(model=resdx.models.CNTModel(), # These inputs should be similar to EnergyPlus model.
                        net_heating_cop_rated=[5],
                        net_heating_capacity_rated=[10600],
                        flow_rated_per_cap_heating_rated=[0.00005],
                        fan_efficacy_heating_rated=[0])
# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np

# Plot integrated power and capacity
T_out = np.arange(-23,21+1,(23+21+1)/365)
conditions = [VerificationModel.make_condition(resdx.HeatingConditions,outdoor=resdx.PsychState(drybulb=resdx.fr_u(T,"°C"),rel_hum=0.5)) for T in T_out] # Same artificial weather used in EnergyPlus.
Q_with_frost = [VerificationModel.gross_heating_capacity_with_frost(condition) for condition in conditions]
Q_defrost_indoor = [VerificationModel.gross_heating_capacity_defrost_indoor(condition) for condition in conditions]
#COP_integrated = [VerificationModel.gross_integrated_heating_cop(condition) for condition in conditions]

# To create a standalone figure window
%matplotlib auto

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Temp (°C)')
ax1.set_ylabel('Capacity with frost (W)', color=color)
ax1.plot(T_out, Q_with_frost, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([9000,11000])

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Defrost -indoor- capacity (W)', color=color)
ax2.plot(T_out, Q_defrost_indoor, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([-10,4000])

fig.tight_layout()
plt.savefig('output/verification-model-heat-pump.png')

# %%

#%%
import resdx

# This example compares just the defrost characteristics (everything else is held constant)

class DefrostVerificationModel(resdx.models.ConstantDXModel):

  @staticmethod
  def gross_integrated_heating_capacity(conditions, system):
    return resdx.models.NRELDXModel.gross_integrated_heating_capacity(conditions, system)

  @staticmethod
  def gross_integrated_heating_power(conditions, system):
    return resdx.models.NRELDXModel.gross_integrated_heating_power(conditions, system)


test_dx_system = resdx.DXUnit(model=DefrostVerificationModel(), # These inputs should be similar to EnergyPlus model.
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
T_out = np.arange(-23,22,(23+22)/365)
conditions = [test_dx_system.make_condition(resdx.HeatingConditions,outdoor=resdx.PsychState(drybulb=resdx.fr_u(T,"°C"),rel_hum=0.4)) for T in T_out] # Same artificial weather used in EnergyPlus.
Q = [test_dx_system.gross_integrated_heating_capacity(condition) for condition in conditions]

# To create a standalone figure window

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Temp (°C)')
ax1.set_ylabel('Integrated Capacity (W)', color=color)
ax1.plot(T_out, Q, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([7000,11000])

fig.tight_layout()
plt.savefig('output/verification-model-heat-pump.png')

# %%

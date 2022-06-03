#%%
import resdx
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def plot(x, y, xlabel, ylabel,figure_name):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(1.,8.)
    plt.savefig(f'output/{figure_name}.png')

# Cooling
seer_range = np.arange(6,26.5,0.5)

# Heating
hspf_range = np.arange(5,16,0.5)

#%% Cooling inverse calculations with constraints (single speed)
cops_from_seer = []
for seer in seer_range:
    root_fn = lambda x : resdx.DXUnit(rated_gross_cooling_cop=x, input_seer=seer).seer() - seer
    cop, solution = optimize.newton(root_fn,seer/3.33, full_output = True)
    #print(f"seer = {seer:.1f}, cop = {cop:.2f}, fan_eff = {resdx.to_u(fan_efficacy(seer),'W/cfm'):.3f}, C_d = {c_d(seer):.3f}, converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_seer.append(cop)

plot(seer_range, cops_from_seer, "SEER", "Gross COP (at A conditions)","cooling-cop-v-seer")

#%% Cooling inverse calculations with constraints (two speed)
cops_from_seer = []

for seer in seer_range:
    root_fn = lambda x : resdx.DXUnit(
        number_of_input_stages = 2, rated_gross_cooling_cop=x, input_seer=seer).seer() - seer
    cop, solution = optimize.newton(root_fn,seer/3.33, full_output = True)
    #print(f"seer = {seer:.1f}, cop = {cop:.2f}, fan_eff = {resdx.to_u(fan_efficacy(seer),'W/cfm'):.3f}, C_d = {c_d(seer):.3f}, converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_seer.append(cop)

plot(seer_range, cops_from_seer, "SEER", "Gross COP (at A conditions)","cooling-2-cop-v-seer")

#%% Heating inverse calculations (single speed)
cops_from_hspf = []
for hspf in hspf_range:
    root_fn = lambda x : resdx.DXUnit(rated_gross_heating_cop=x, input_hspf=hspf).hspf() - hspf
    cop, solution = optimize.newton(root_fn,hspf/2.0, full_output = True)
    #print(f"hspf = {hspf:.1f}, cop = {cop:.2f}, fan_eff = {resdx.to_u(fan_efficacy(estimated_seer(hspf)),'W/cfm'):.3f}, C_d = {c_d(estimated_seer(hspf)):.3f},converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_hspf.append(cop)

plot(hspf_range, cops_from_hspf, "HSPF", "Gross COP (at H1 conditions)","heating-cop-v-hspf")

#%% Heating inverse calculations (two speed)
cops_from_hspf = []
for hspf in hspf_range:
    root_fn = lambda x : resdx.DXUnit(
        number_of_input_stages=2, rated_gross_heating_cop=x, input_hspf=hspf).hspf() - hspf
    cop, solution = optimize.newton(root_fn,hspf/2.0, full_output = True)
    #print(f"hspf = {hspf:.1f}, cop = {cop:.2f}, fan_eff = {resdx.to_u(fan_efficacy(estimated_seer(hspf)),'W/cfm'):.3f}, C_d = {c_d(estimated_seer(hspf)):.3f},converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_hspf.append(cop)

plot(hspf_range, cops_from_hspf, "HSPF", "Gross COP (at H1 conditions)","heating-2-cop-v-hspf")

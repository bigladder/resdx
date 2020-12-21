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
    plt.savefig(f'output/{figure_name}.png')

# Cooling 

def fan_efficacy(seer):
    if seer <= 14:
        return resdx.u(0.25,'W/(cu_ft/min)')
    elif seer >= 16:
        return resdx.u(0.18,'W/(cu_ft/min)')
    else:
        return resdx.u(0.25,'W/(cu_ft/min)') + (resdx.u(0.18,'W/(cu_ft/min)') - resdx. u(0.25,'W/(cu_ft/min)'))/2.0 * (seer - 14.0) 


def c_d(seer):
    if seer <= 12:
        return 0.2
    elif seer >= 13:
        return 0.1
    else:
        return 0.2 + (0.1 - 0.2)*(seer - 12.0)

# Heating

def estimated_seer(hspf): # Linear model fitted (R² = 0.994) based on data of the histrory of federal minimums (https://www.eia.gov/todayinenergy/detail.php?id=40232#).
    return (hspf - 3.2627)/0.3526

#%% Cooling inverse calculations with constraints
seer_range = np.arange(6,26.5,0.1)
cops_from_seer = []
for seer in seer_range:
    root_fn = lambda x : resdx.DXUnit(gross_cooling_cop_rated = [x], fan_eff_cooling_rated = [fan_efficacy(seer)], c_d_cooling=c_d(seer)).seer() - seer
    cop, solution = optimize.newton(root_fn,seer/3.33, full_output = True)
    #print(f"seer = {seer:.1f}, cop = {cop:.2f}, fan_eff = {resdx.convert(fan_efficacy(seer),'W/(m**3/s)','W/(cu_ft/min)'):.3f}, C_d = {c_d(seer):.3f}, converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_seer.append(cop)

plot(seer_range, cops_from_seer, "SEER", "Gross COP (at A conditions)","cooling-cop-v-seer")
# seer = 13.0, cop = 3.72, fan_eff = 0.250, C_d = 0.100,
#%% Heating inverse calculations
hspf_range = np.arange(5,14,0.1)
cops_from_hspf = []
for hspf in hspf_range:
    root_fn = lambda x : resdx.DXUnit(gross_heating_cop_rated = [x], fan_eff_heating_rated = [fan_efficacy(estimated_seer(hspf))], c_d_heating=c_d(estimated_seer(hspf))).hspf() - hspf
    cop, solution = optimize.newton(root_fn,hspf/2.0, full_output = True)
    #print(f"hspf = {hspf:.1f}, cop = {cop:.2f}, fan_eff = {resdx.convert(fan_efficacy(estimated_seer(hspf)),'W/(m**3/s)','W/(cu_ft/min)'):.3f}, C_d = {c_d(estimated_seer(hspf)):.3f},converged = {solution.converged}, iter = {solution.iterations}")
    cops_from_hspf.append(cop)

plot(hspf_range, cops_from_hspf, "HSPF", "Gross COP (at H1 conditions)","heating-cop-v-hspf")
# hspf = 7.7, cop = 3.88, fan_eff = 0.250, C_d = 0.142

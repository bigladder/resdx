#%%
import resdx
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot(x, y, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('../output/'+ylabel.replace(" ", "")+'_vs_'+xlabel.replace(" ", "")+'.png')


def fan_efficacy(seer):
    if seer <= 14:
        return resdx.u(0.25,'W/(cu_ft/min)')
    elif seer >= 16:
        return resdx.u(0.18,'W/(cu_ft/min)')
    else:
        return resdx.u(0.25,'W/(cu_ft/min)') + (resdx.u(0.18,'W/(cu_ft/min)') - resdx. u(0.25,'W/(cu_ft/min)'))/2.0 * (seer - 14.0) 


def C_d(seer):
    if seer <= 12:
        return 0.25
    elif seer >= 13:
        return 0.1
    else:
        return 0.25 + (0.1 - 0.25)*(seer - 12.0)

#%% Cooling inverse calculations with constraints
seer_range = np.arange(6,26.5,0.1)
cops = []
for seer in seer_range:
    root_fn = lambda x : resdx.DXUnit(gross_cooling_cop_rated = [x], fan_eff_cooling_rated = [fan_efficacy(seer)], c_d_cooling=C_d(seer)).seer() - seer
    cop, solution = optimize.newton(root_fn,seer/3.33, full_output = True)
    print(f"seer = {seer:.1f}, cop = {cop:.2f}, fan_eff = {resdx.convert(fan_efficacy(seer),'W/(m**3/s)','W/(cu_ft/min)'):.3f}, C_d = {C_d(seer):.3f}, converged = {solution.converged}, iter = {solution.iterations}")
    cops.append(cop)

plot(seer_range, cops, "SEER", "Gross COP")

#%% Heating inverse calculations
hspf_range = np.arange(5,14,0.1)
cops = []
for hspf in hspf_range:
    root_fn = lambda x : resdx.DXUnit(gross_heating_cop_rated = [x]).hspf() - hspf
    cop, solution = optimize.newton(root_fn,hspf/2.0, full_output = True)
    print(f"hspf = {hspf:.1f}, cop = {cop:.2f}, converged = {solution.converged}, iter = {solution.iterations}")
    cops.append(cop)

plot(hspf_range, cops, "HSPF", "Gross COP")
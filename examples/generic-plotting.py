
#%%
import resdx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

import numpy as np
#%% Plotting function class
def makeplots(ConditionsArray,FunctionsArray,FunctionsArray_2nd_axis = None):
    # create color shades of Red
    norm = mpl.colors.Normalize(vmin=0, vmax=len(ConditionsArray.secondary_independent_values[0]))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds) # greens
    # create figure and set primary variable label for the x-axis
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(f"{ConditionsArray.primary_variable_name} [{ConditionsArray.primary_variable_unit}]")

    plotting(fig,ax1,ConditionsArray,FunctionsArray,cmap)
    ax1.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3,fancybox=True, shadow=True)
    ax1.tick_params(axis='y', labelcolor='red')

    if FunctionsArray_2nd_axis is not None:
        # create color shades of Green
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Greens) # Reds

        ax2 = ax1.twinx()

        plotting(fig,ax2,ConditionsArray,FunctionsArray_2nd_axis,cmap)
        ax2.legend(bbox_to_anchor=(0,-0.2,1,0.2), loc='lower left', mode="expand", borderaxespad=0, ncol=3,fancybox=True, shadow=True)
        ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    # plt.savefig('output/generic-plotting.png')
    return

def plotting(fig,ax,ConditionsArray,FunctionsArray,cmap):
    linestyles = ["solid","dashed","dotted","dashdot"] # for functions on 2 different y-axis
    markers_n_colors = gen_combinations(ConditionsArray, markers_n_colors = True) # all combinations of: (color,marker)
    combin_values = gen_combinations(ConditionsArray) # all combinations of secondary variables values: (first secondary variable, second secondary variable)

    models_names = ""
    for i in np.arange(0,len(FunctionsArray.labels)):
        models_names_temp = FunctionsArray.labels[i] + '=' + linestyles[i] + " | "
        models_names = models_names + models_names_temp

    ax.set_ylabel(f"{FunctionsArray.functions_unique_name} [{FunctionsArray.functions_unique_unit}] | {models_names}")
    markers_n_colors_count = 0
    for cond_l in ConditionsArray.conditions_all_combinations:
        linestyle_count = 0
        fun_count = 0 # to only show one model label per y-axis.
        for fun in FunctionsArray.functions:
            x = ConditionsArray.primary_independent_values #(maybe you should check they are in order?)
            y = [resdx.to_u(fun(cond),FunctionsArray.functions_unique_unit) for cond in cond_l]
            if fun_count > 0:
                label = None
            else: # for now only 2 secondary variables are accepted
                label = f"{ConditionsArray.secondary_variables_names[0]}={combin_values[markers_n_colors_count][0]} [{ConditionsArray.secondary_variables_units[0]}], {ConditionsArray.secondary_variables_names[1]}={combin_values[markers_n_colors_count][1]} [{ConditionsArray.secondary_variables_units[1]}]"

            ax.plot(x, y, color=cmap.to_rgba(1+markers_n_colors[markers_n_colors_count][0]), marker=markers_n_colors[markers_n_colors_count][1], linestyle=linestyles[linestyle_count], label=label)

            fun_count = fun_count + 1
            linestyle_count = linestyle_count + 1
        markers_n_colors_count = markers_n_colors_count + 1
    return
#%% generate markers and colors or secondary variables data combinations
def gen_combinations(ConditionsArray, markers_n_colors = False):
    if markers_n_colors is False:
        iterator = ConditionsArray.secondary_independent_values
    else:
        # colors = ["green","red","blue","orange"] # first secondary variable
        colors = [0,1,2,3] # or colors shades for first secondary variable
        markers = ["o","s","*","+","x","v","^"] # second secondary variable
        first_var_color = colors[0:len(ConditionsArray.secondary_independent_values[0])]
        second_var_marker = markers[0:len(ConditionsArray.secondary_independent_values[1])]

        iterator = [first_var_color,second_var_marker]

    combined_lists = []
    for it in iterator[0]:
        temp_list = list(zip([it]*len(iterator[1]),iterator[1]))
        combined_lists = combined_lists + temp_list
    return combined_lists

#%% Conditions class
class ConditionsArray:

    def __init__(self, primary_independent_values, secondary_independent_values, primary_variable_name, secondary_variables_names,
    primary_variable_unit, secondary_variables_units, conditions_all_combinations = None):
        self.primary_independent_values = primary_independent_values
        self.secondary_independent_values = secondary_independent_values
        self.primary_variable_name = primary_variable_name
        self.secondary_variables_names = secondary_variables_names
        self.primary_variable_unit = primary_variable_unit
        self.secondary_variables_units = secondary_variables_units
        self.conditions_all_combinations = conditions_all_combinations

    def gen_conditions(self):
        ## TODO ##
        return

#%% Mannually create conditions
ConditionsArray.primary_independent_values = [70, 75, 80, 85, 90, 95, 100]
ConditionsArray.secondary_independent_values = [[700, 1125, 2000], [0.001, 0.005, 0.01, 0.015]]
ConditionsArray.primary_variable_name = "Tempearture (db)"
ConditionsArray.secondary_variables_names = ["Flowrate", "Indoor hr"]
ConditionsArray.primary_variable_unit =  "°F"
ConditionsArray.secondary_variables_units = ["cu_ft/min", ""]

#%% All combinations of secondary variables

# Conditions for each secondary variables combination
dxunit = resdx.DXUnit()
combined_lists = gen_combinations(ConditionsArray)

ConditionsArray.conditions_all_combinations = [[dxunit.make_condition(resdx.CoolingConditions,outdoor=resdx.PsychState(drybulb=resdx.fr_u(T,"°F"),hum_rat=0.003),
indoor=resdx.PsychState(drybulb=resdx.fr_u(80,"°F"),hum_rat=second_var[1]),
air_vol_flow=resdx.fr_u(second_var[0],"cu_ft/min")) for T in ConditionsArray.primary_independent_values] for second_var in  combined_lists]

#%% Functions class
class FunctionsArray:

    def __init__(self,functions,labels,functions_unique_name,functions_unique_unit):
        self.functions = functions
        self.labels = labels
        self.functions_unique_name = functions_unique_name
        self.functions_unique_unit = functions_unique_unit

FunctionsArray.functions = [dxunit.gross_total_cooling_capacity, dxunit.net_total_cooling_capacity]
FunctionsArray.labels = ["Gross", "Net"]
FunctionsArray.functions_unique_name = "Total cooling capacity"
FunctionsArray.functions_unique_unit = "W"

functions = [dxunit.gross_cooling_power, dxunit.net_cooling_power]
labels = ["Gross", "Net"]
functions_unique_name = "Total cooling power"
functions_unique_unit = "W"
FunctionsArray_2nd_axis = FunctionsArray(functions,labels,functions_unique_name,functions_unique_unit)
#%% plot example
%matplotlib auto
makeplots(ConditionsArray,FunctionsArray,FunctionsArray_2nd_axis = None)
makeplots(ConditionsArray,FunctionsArray,FunctionsArray_2nd_axis)
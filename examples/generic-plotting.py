
#%%
import resdx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()

import numpy as np

dxunit = resdx.DXUnit()

# Make dimensions (user defined)
drybulb_temperatures = resdx.IndependentDimension(values= [70, 75, 80, 85, 90, 95, 100],
                                                  name="Outdoor Drybulb Temperature",
                                                  units="°F")
flow_rates = resdx.IndependentDimension(values=[700, 1125, 2000],
                                        name="Volumetric Flow Rate",
                                        units="cu_ft/min")
indoor_hr = resdx.IndependentDimension(values=[0.001, 0.005, 0.01, 0.015],
                                       name="Indoor hr",
                                       units="")

# Make conditions (user defined...for now)
conditions_matrix = []
for combined_values in dxunit.independent_values_combinations(flow_rates.values,indoor_hr.values): # the user can choose which variables to consider here
    conditions_matrix_temporary = []
    for tdb in drybulb_temperatures.values: # the user can choose which variable to be on the x-axis
        conditions_matrix_temporary.append(dxunit.make_condition(resdx.CoolingConditions,
                                                            outdoor=resdx.PsychState(drybulb=resdx.fr_u(tdb,"°F"),
                                                            hum_rat=0.003),
                                                            indoor=resdx.PsychState(drybulb=resdx.fr_u(80,"°F"),
                                                            hum_rat=combined_values[1]),
                                                            air_vol_flow=resdx.fr_u(combined_values[0],"cu_ft/min")))
    conditions_matrix.append(conditions_matrix_temporary)

conditions_array = resdx.ConditionsArray(independent_dimensions = [drybulb_temperatures, flow_rates, indoor_hr],
                                         conditions_matrix = conditions_matrix)

# Make functions. Only 2 function arrays are allowed. one function category per y-axis (functions_array for y left and functions_array2 for y right)
functions_array =  resdx.FunctionsArray(functions = [dxunit.gross_total_cooling_capacity, dxunit.net_total_cooling_capacity,dxunit.gross_cooling_power,dxunit.net_cooling_power], # Functions in each function array should have the same unit/physical meaning
                                        labels = ["Gross capacity", "Net capacity","Gross power", "Net power"],
                                        functions_unique_name = "Total cooling capacity/Power",
                                        functions_unique_unit = "W")

functions_array2 =  resdx.FunctionsArray(functions = [dxunit.net_cooling_cop, dxunit.gross_cooling_cop],
                                        labels = ["Net", "Gross"],
                                        functions_unique_name = "Cooling COP",
                                        functions_unique_unit = "")

# plot
%matplotlib auto
dxunit.make_plot(conditions_array,functions_array,functions_array2)
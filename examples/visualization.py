import resdx
from koozie import fr_u
from dimes.griddeddata import GridAxis, GridPointData, RegularGridData, DataSelection
from dimes import DimensionalPlot, DimensionalData, DisplayData, LinesOnly
from resdx.util import geometric_space

output_directory_path = "output"

Q95rated = 14000
Q47rated = 18000
seer2 = 21.0
eer2 = 13.0
hspf2 = 11.0

dx_unit = resdx.DXUnit(
    staging_type=resdx.StagingType.VARIABLE_SPEED,
    rated_net_heating_capacity=fr_u(Q47rated, "Btu/h"),
    rated_net_total_cooling_capacity=fr_u(Q95rated, "Btu/h"),
    input_seer=seer2,
    input_eer=eer2,
    input_hspf=hspf2,
)

# Indoor conditions variability

cop_values = []
indoor_temperatures = GridAxis(
    [65.0, 67.5, 70.0, 72.5, 75.0], "Indoor Temperature", "degF"
)
indoor_airflow_rates = GridAxis(
    [300.0, 350.0, 400.0, 450.0], "Indoor Airflow Rate", "cfm/ton"
)
for indoor_temperature in indoor_temperatures.data_values:
    for air_flow in indoor_airflow_rates.data_values:
        condition = dx_unit.make_condition(
            resdx.HeatingConditions,
            compressor_speed=dx_unit.heating_full_load_speed,
            indoor=resdx.PsychState(
                drybulb=fr_u(indoor_temperature, "degF"), rel_hum=0.4
            ),
        )
        condition.set_mass_airflow_ratio(air_flow / 400.0)
        cop_values.append(dx_unit.net_steady_state_heating_cop(condition))

cops = GridPointData(cop_values, "Net Heating COP (at H1 full conditions)", "")

plot = RegularGridData([indoor_temperatures, indoor_airflow_rates], [cops]).make_plot(
    x_grid_axis=DataSelection("Indoor Temperature"),
    legend_grid_axis=DataSelection("Indoor Airflow Rate", precision=0),
)

plot.write_html_plot(f"{output_directory_path}/indoor_conditions.html")

# SHR

shr_values = []
for air_flow in indoor_airflow_rates.data_values:
    condition = dx_unit.make_condition(
        resdx.CoolingConditions,
        compressor_speed=dx_unit.heating_full_load_speed,
    )
    condition.set_mass_airflow_ratio(air_flow / 400.0)
    shr_values.append(dx_unit.gross_shr(condition))

shrs = DisplayData(shr_values, "Rated SHR", "", line_properties=LinesOnly())


plot = DimensionalPlot(indoor_airflow_rates)

plot.add_display_data(shrs)

plot.write_html_plot(f"{output_directory_path}/shr.html")

# Heating temperature bins
distributions = resdx.DXUnit.regional_heating_distributions[
    resdx.AHRIVersion.AHRI_210_240_2023
]
regions = GridAxis([1, 2, 3, 4, 5, 6], "Region", "")
dry_bulbs = GridAxis(
    resdx.HeatingDistribution.outdoor_drybulbs, "Outdoor Temperature", "K"
)
fraction_values = []
for region in regions.data_values:
    for index, _ in enumerate(dry_bulbs.data_values):
        fraction_values.append(distributions[region].fractional_hours[index])

fractions = GridPointData(fraction_values, "Fraction", "")
plot = RegularGridData([regions, dry_bulbs], [fractions]).make_plot(
    x_grid_axis=DataSelection("Outdoor Temperature", "degF"),
    legend_grid_axis=DataSelection("Region", precision=0),
)

plot.write_html_plot(f"{output_directory_path}/heating_temperature_bins.html")

# Defrost degradation
outdoor_temperatures = DimensionalData(
    data_values=fr_u(
        geometric_space(5.0, 50.0, 45),
        "degF",
    ),
    name="Outdoor Temperature",
    native_units="K",
    display_units="degF",
)

capacity_degradation = DisplayData(
    [
        1.0
        - resdx.models.unified_resnet.RESNETDXModel.defrost_capacity_multiplier(
            outdoor_temperature
        )
        for outdoor_temperature in outdoor_temperatures.data_values
    ],
    "Capacity",
    "",
    y_axis_name="Defrost Degradation Factor",
    line_properties=LinesOnly(),
)

power_degradation = DisplayData(
    [
        1.0
        - resdx.models.unified_resnet.RESNETDXModel.defrost_power_multiplier(
            outdoor_temperature
        )
        for outdoor_temperature in outdoor_temperatures.data_values
    ],
    "Power",
    "",
    y_axis_name="Defrost Degradation Factor",
    line_properties=LinesOnly(),
)

plot = DimensionalPlot(outdoor_temperatures)

plot.add_display_data(capacity_degradation)

plot.add_display_data(power_degradation)

plot.write_html_plot(f"{output_directory_path}/defrost_degradation.html")


# Cycling degradation
def cycling_multiplier(cycling_ratio, cycling_degradation):
    return 1.0 / (1.0 + cycling_degradation * (cycling_ratio - 1.0))


cycling_ratios = DimensionalData(
    data_values=geometric_space(0.0, 1.0, 10),
    name="Cycling Ratio",
    native_units="",
    display_units="",
)

variable_speed_degradation = DisplayData(
    data_values=[
        cycling_multiplier(cycling_ratio, 0.4) - 1.0
        for cycling_ratio in cycling_ratios.data_values
    ],
    name="Variable Speed",
    native_units="",
    display_units="%",
    y_axis_name="Increased Energy Consumption",
    line_properties=LinesOnly(),
)

single_speed_degradation = DisplayData(
    data_values=[
        cycling_multiplier(cycling_ratio, 0.08) - 1.0
        for cycling_ratio in cycling_ratios.data_values
    ],
    name="Single or Two Stage",
    native_units="",
    display_units="%",
    y_axis_name="Increased Energy Consumption",
    line_properties=LinesOnly(),
)

plot = DimensionalPlot(cycling_ratios)

plot.add_display_data(variable_speed_degradation)

plot.add_display_data(single_speed_degradation)

plot.write_html_plot(f"{output_directory_path}/cycling_degradation.html")

# Fan power

air_flow_ratios = DimensionalData(geometric_space(0.0, 1.2, 20), "Airflow Ratio", "")

ducted_bpm_ratio = DisplayData(
    [x**2.75 for x in air_flow_ratios.data_values],
    "BPM Motor (Ducted System)",
    "",
    y_axis_name="Power Ratio",
    line_properties=LinesOnly(),
)

ductless_bpm_ratio = DisplayData(
    [x**3 for x in air_flow_ratios.data_values],
    "BPM Motor (Ductless System)",
    "",
    y_axis_name="Power Ratio",
    line_properties=LinesOnly(),
)

psc_ratio = DisplayData(
    [x * (0.3 * x + 0.7) for x in air_flow_ratios.data_values],
    "PSC Motor",
    "",
    y_axis_name="Power Ratio",
    line_properties=LinesOnly(),
)

plot = DimensionalPlot(air_flow_ratios)

plot.add_display_data(ducted_bpm_ratio)

plot.add_display_data(ductless_bpm_ratio)

plot.add_display_data(psc_ratio)

plot.write_html_plot(f"{output_directory_path}/fan_power_ratios.html")

ducted_bpm_sfp = DisplayData(
    [0.281 * x**1.75 for x in air_flow_ratios.data_values],
    "BPM Motor (Ducted System)",
    "W/cfm",
    y_axis_name="Specific Fan Power",
    line_properties=LinesOnly(),
)

ductless_bpm_sfp = DisplayData(
    [0.171 * x**2 for x in air_flow_ratios.data_values],
    "BPM Motor (Ductless System)",
    "W/cfm",
    y_axis_name="Specific Fan Power",
    line_properties=LinesOnly(),
)

psc_sfp = DisplayData(
    [0.414 * (0.3 * x + 0.7) for x in air_flow_ratios.data_values],
    "PSC Motor",
    "W/cfm",
    y_axis_name="Specific Fan Power",
    line_properties=LinesOnly(),
)

plot = DimensionalPlot(air_flow_ratios)

plot.add_display_data(ducted_bpm_sfp)

plot.add_display_data(ductless_bpm_sfp)

plot.add_display_data(psc_sfp)

plot.write_html_plot(f"{output_directory_path}/specific_fan_power.html")

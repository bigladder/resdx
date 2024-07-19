# %%
import numpy as np
from scipy import optimize

from koozie import fr_u

from dimes import MarkersOnly, LinesOnly, DimensionalPlot, DisplayData, DimensionalData


from resdx.util import (
    linear,
    linear_string,
    quadratic,
    quadratic_string,
    cubic,
    cubic_string,
    quartic,
    quartic_string,
    calculate_r_squared,
)

import resdx


def plot(x, ys, y_axis_name, figure_name):
    plot = DimensionalPlot(x)
    for y in ys:
        plot.add_display_data(y, axis_name=y_axis_name)
    plot.write_html_plot(f"output/{figure_name}.html")


def make_objective_function(comparison, target):
    return lambda x: comparison(x, target) - target


def get_inverse_values(
    target_range,
    metric_calculation,
    initial_guess=lambda x: x / 3.0,
    curve_fit_function=quadratic,
    curve_fit_guesses=(1, 1, 1),
):
    inverse_values = []
    for target in target_range.data_values:
        print(f"    {target_range.name}={target:.2f}")
        try:
            inverse_values.append(
                optimize.newton(
                    make_objective_function(metric_calculation, target),
                    initial_guess(target),
                )
            )
        except RuntimeError:
            raise RuntimeError(
                f"Unable to find solution for target: {target_range.name}={target}."
            )
    print(f"    curve fitting...")
    curve_fit_coefficients = optimize.curve_fit(
        curve_fit_function, target_range.data_values, inverse_values, curve_fit_guesses
    )[0]
    return inverse_values, curve_fit_coefficients


def make_plot(target_range, systems, axis_title, figure_name):
    display_data = []
    for system_name, system in systems.items():
        print(f"Calculating system: {system_name}")
        try:
            inputs, coefficients = get_inverse_values(
                target_range,
                system["calculation"],
                initial_guess=system["initial_guess"],
                curve_fit_function=system["curve_fit"]["function"],
                curve_fit_guesses=system["curve_fit"]["guesses"],
            )
        except RuntimeError as e:
            raise RuntimeError(f"Unable to find solution for {system_name}: {e}")

        display_data.append(
            DisplayData(
                inputs,
                name=system_name,
                native_units="W/W",
                line_properties=MarkersOnly(),
            )
        )
        curve_fit_string = system["curve_fit"]["string"](
            target_range.name, *coefficients
        )
        curve_fit_data = [
            system["curve_fit"]["function"](metric, *coefficients)
            for metric in target_range.data_values
        ]
        r2 = calculate_r_squared(inputs, curve_fit_data)
        display_data.append(
            DisplayData(
                curve_fit_data,
                name=f"{system_name}: {curve_fit_string}, R2={r2:.4g}",
                native_units="W/W",
                line_properties=LinesOnly(),
            )
        )

    plot(target_range, display_data, axis_title, figure_name)


# Cooling
seer_eer_ratio_range = np.linspace(1.2, 2.0, 10)

ts_cooling_systems = {}
vs_cooling_systems = {}

for seer_eer_ratio in seer_eer_ratio_range:
    ts_cooling_systems[f"Two Speed (SEER2/EER2={seer_eer_ratio:.2})"] = {
        "calculation": lambda cop_82_min, seer, ratio=seer_eer_ratio: resdx.DXUnit(
            staging_type=resdx.StagingType.TWO_STAGE,
            rated_net_total_cooling_cop_82_min=cop_82_min,
            input_eer=seer / ratio,
            input_seer=seer,
            input_hspf=10.0,
        ).seer(),
        "initial_guess": lambda x: x / 3.0,
        "curve_fit": {
            "function": linear,
            "string": linear_string,
            "guesses": (1, 1),
        },
    }

    vs_cooling_systems[f"Variable Speed (SEER2/EER2={seer_eer_ratio:.2})"] = {
        "calculation": lambda cop_82_min, seer, ratio=seer_eer_ratio: resdx.DXUnit(
            staging_type=resdx.StagingType.VARIABLE_SPEED,
            rated_net_total_cooling_cop_82_min=cop_82_min,
            input_eer=seer / ratio,
            input_seer=seer,
            input_hspf=10.0,
        ).seer(),
        "initial_guess": lambda x: x / 3.0,
        "curve_fit": {
            "function": linear,
            "string": linear_string,
            "guesses": (1, 1),
        },
    }


make_plot(
    DimensionalData(
        np.linspace(6, 26.5, 2), name="SEER2", native_units="Btu/Wh"
    ),  # All straight lines don't need more than two points
    ts_cooling_systems,
    "Net COP (at B low conditions)",
    "cooling-2s-cop-v-seer",
)


make_plot(
    DimensionalData(
        np.linspace(14, 35, 3), name="SEER2", native_units="Btu/Wh"
    ),  # All straight lines don't need more than two points
    vs_cooling_systems,
    "Net COP (at B low conditions)",
    "cooling-vs-eir_r-v-seer",
)

cap_17_maintenance_range = [0.5, 0.55, 0.6, 0.7, 0.8, 1.1]

ss_heating_systems = {}
ts_heating_systems = {}
vs_heating_systems = {}

for cap_17_maintenance in cap_17_maintenance_range:
    ss_heating_systems[f"Single Speed (Q17/Q47={cap_17_maintenance:.2})"] = {
        "calculation": lambda cop, hspf, cap_m=cap_17_maintenance: resdx.DXUnit(
            staging_type=resdx.StagingType.SINGLE_STAGE,
            rated_net_heating_capacity=fr_u(3.0, "ton_ref"),
            rated_net_heating_capacity_17=fr_u(3.0, "ton_ref") * cap_m,
            rated_net_heating_cop=cop,
            input_eer=10,
            input_seer=19.0,
            input_hspf=hspf,
        ).hspf(),
        "initial_guess": lambda x: x / 2.0,
        "curve_fit": {
            "function": quadratic,
            "string": quadratic_string,
            "guesses": (1, 1, 1),
        },
    }

    ts_heating_systems[f"Two Speed (Q17/Q47={cap_17_maintenance:.2})"] = {
        "calculation": lambda cop, hspf, cap_m=cap_17_maintenance: resdx.DXUnit(
            staging_type=resdx.StagingType.TWO_STAGE,
            rated_net_heating_capacity=fr_u(3.0, "ton_ref"),
            rated_net_heating_capacity_17=fr_u(3.0, "ton_ref") * cap_m,
            rated_net_heating_cop=cop,
            input_eer=10,
            input_seer=19.0,
            input_hspf=hspf,
        ).hspf(),
        "initial_guess": lambda x: x / 2.0,
        "curve_fit": {
            "function": quadratic,
            "string": quadratic_string,
            "guesses": (1, 1, 1),
        },
    }

    vs_heating_systems[f"Variable Speed (Q17/Q47={cap_17_maintenance:.2})"] = {
        "calculation": lambda cop, hspf, cap_m=cap_17_maintenance: resdx.DXUnit(
            staging_type=resdx.StagingType.VARIABLE_SPEED,
            rated_net_heating_capacity=fr_u(3.0, "ton_ref"),
            rated_net_heating_capacity_17=fr_u(3.0, "ton_ref") * cap_m,
            rated_net_heating_cop=cop,
            input_eer=10,
            input_seer=19.0,
            input_hspf=hspf,
        ).hspf(),
        "initial_guess": lambda x: x / 2.0,
        "curve_fit": {
            "function": quadratic,
            "string": quadratic_string,
            "guesses": (1, 1, 1),
        },
    }

hspf_range = DimensionalData(np.linspace(5, 11, 5), name="HSPF2", native_units="Btu/Wh")


make_plot(
    hspf_range,
    ss_heating_systems,
    "Net COP (at H1 full conditions)",
    "heating-1s-cop-v-hspf",
)

make_plot(
    hspf_range,
    ts_heating_systems,
    "Net COP (at H1 full conditions)",
    "heating-2s-cop-v-hspf",
)

hspf_range = DimensionalData(np.linspace(5, 16, 5), name="HSPF2", native_units="Btu/Wh")


make_plot(
    hspf_range,
    vs_heating_systems,
    "Net COP (at H1 full conditions)",
    "heating-vs-cop-v-hspf",
)

# %%
import csv
from copy import deepcopy
import numpy as np
from scipy import optimize

from koozie import fr_u

from dimes import MarkersOnly, LinesOnly, DimensionalPlot, DisplayData, DimensionalData


from resdx.util import (
    linear,
    linear_string,
    quadratic,
    quadratic_string,
    # cubic,
    # cubic_string,
    # quartic,
    # quartic_string,
    calculate_r_squared,
)

import resdx


class CurveFit:
    def __init__(self, function, regression_string, initial_coefficient_guesses):
        self.function = function
        self.regression_string = regression_string
        self.initial_coefficient_guesses = initial_coefficient_guesses


linear_curve_fit = CurveFit(linear, linear_string, (1, 1))
quadratic_curve_fit = CurveFit(quadratic, quadratic_string, (1, 1, 1))


class RatingRegression:
    def __init__(
        self,
        staging_type: resdx.StagingType,
        calculation,
        target_title: str,
        initial_guess,
        rating_range: DisplayData,
        secondary_range: DisplayData,
        curve_fit: CurveFit,
    ):
        self.staging_type = staging_type
        self.calculation = calculation
        self.target_title = target_title
        self.initial_guess = initial_guess
        self.rating_range = rating_range
        self.secondary_range = secondary_range
        self.curve_fit = curve_fit

    def evaluate(self, output_name):
        display_data = []
        for secondary_value in self.secondary_range.data_values:
            series_name = f"{self.secondary_range.name}={secondary_value:.2}"

            print(f"Evaluating {self.staging_type.name} ({series_name})")
            try:
                inputs, coefficients = get_inverse_values(
                    self.rating_range,
                    lambda x, target: self.calculation(
                        x, target, self.staging_type, secondary_value
                    ),
                    self.initial_guess,
                    curve_fit_function=self.curve_fit.function,
                    curve_fit_guesses=self.curve_fit.initial_coefficient_guesses,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Unable to find solution for {self.staging_type.name} ({series_name}): {e}"
                )

            display_data.append(
                DisplayData(
                    inputs,
                    name=series_name,
                    native_units="W/W",
                    line_properties=MarkersOnly(),
                )
            )
            curve_fit_string = self.curve_fit.regression_string(
                self.rating_range.name, *coefficients
            )
            curve_fit_data = [
                self.curve_fit.function(rating, *coefficients)
                for rating in self.rating_range.data_values
            ]
            r2 = calculate_r_squared(inputs, curve_fit_data)
            display_data.append(
                DisplayData(
                    curve_fit_data,
                    name=f"{series_name}: {curve_fit_string}, R2={r2:.4g}",
                    native_units="W/W",
                    line_properties=LinesOnly(),
                )
            )

        plot(self.rating_range, display_data, self.target_title, output_name)


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


# Cooling
def seer_function(cop_82_min, seer, staging_type, seer_eer_ratio):
    return resdx.DXUnit(
        staging_type=staging_type,
        input_seer=seer,
        input_eer=seer / seer_eer_ratio,
        rated_net_total_cooling_cop_82_min=cop_82_min,
        input_hspf=10.0,
    ).seer()


two_speed_cooling_regression = RatingRegression(
    staging_type=resdx.StagingType.TWO_STAGE,
    calculation=seer_function,
    target_title="Net COP (at B low conditions)",
    initial_guess=lambda target: target / 3.0,
    rating_range=DimensionalData(
        np.linspace(6, 26.5, 2), name="SEER2", native_units="Btu/Wh"
    ),  # All straight lines don't need more than two points
    secondary_range=DimensionalData(
        np.linspace(1.2, 2.0, 10), name="SEER2/EER2", native_units="W/W"
    ),
    curve_fit=linear_curve_fit,
)

variable_speed_cooling_regression = deepcopy(two_speed_cooling_regression)
variable_speed_cooling_regression.staging_type = resdx.StagingType.VARIABLE_SPEED
variable_speed_cooling_regression.rating_range = DimensionalData(
    np.linspace(14, 35, 3), name="SEER2", native_units="Btu/Wh"
)

# two_speed_cooling_regression.evaluate("cooling-two-speed-cop82-v-seer")
# variable_speed_cooling_regression.evaluate("cooling-variable-speed-cop82-v-seer")


# Heating
def hspf_function(cop_47, hspf, staging_type, cap17m):
    return resdx.DXUnit(
        staging_type=staging_type,
        rated_net_heating_capacity=fr_u(3.0, "ton_ref"),
        rated_net_heating_capacity_17=fr_u(3.0, "ton_ref") * cap17m,
        rated_net_heating_cop=cop_47,
        input_hspf=hspf,
        input_seer=19.0,
        input_eer=10.0,
    ).hspf()


single_speed_heating_regression = RatingRegression(
    staging_type=resdx.StagingType.SINGLE_STAGE,
    calculation=hspf_function,
    target_title="Net COP (at H1 full conditions)",
    initial_guess=lambda target: target / 2.0,
    rating_range=DimensionalData(
        np.linspace(5, 11, 5), name="HSPF2", native_units="Btu/Wh"
    ),
    secondary_range=DimensionalData(
        [0.5, 0.55, 0.6, 0.7, 0.8, 1.1], name="Q17/Q47", native_units="Btu/Btu"
    ),
    curve_fit=quadratic_curve_fit,
)

two_speed_heating_regression = deepcopy(single_speed_heating_regression)
two_speed_heating_regression.staging_type = resdx.StagingType.TWO_STAGE

variable_speed_heating_regression = deepcopy(single_speed_heating_regression)
variable_speed_heating_regression.staging_type = resdx.StagingType.VARIABLE_SPEED
variable_speed_cooling_regression.rating_range = DimensionalData(
    np.linspace(5, 16, 5), name="HSPF2", native_units="Btu/Wh"
)

single_speed_heating_regression.evaluate("heating-single-speed-cop47-v-hspf")
two_speed_heating_regression.evaluate("heating-two-speed-cop47-v-hspf")
variable_speed_heating_regression.evaluate("heating-variable-speed-cop47-v-hspf")

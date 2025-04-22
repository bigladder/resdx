# %%
import csv
from copy import deepcopy
from typing import Dict, List
from pathlib import Path
import time

from numpy import linspace
from scipy import optimize
from jinja2 import Environment, FileSystemLoader

from koozie import fr_u

from dimes import (
    LineProperties,
    MarkersOnly,
    LinesOnly,
    DimensionalPlot,
    DisplayData,
    DimensionalData,
)


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
    geometric_space,
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

    input_data: List[List[float]]

    def __init__(
        self,
        staging_type: resdx.StagingType,
        calculation,
        input_title: str,
        initial_guess,
        rating_range: DisplayData,
        secondary_range: DisplayData,
        curve_fit: CurveFit,
    ):
        self.staging_type = staging_type
        self.calculation = calculation
        self.input_title = input_title
        self.initial_guess = initial_guess
        self.rating_range = rating_range
        self.secondary_range = secondary_range
        self.curve_fit = curve_fit

    def evaluate(self, output_name, do_curve_fit=False):
        display_data = []
        csv_output_data = {
            self.rating_range.name: [],
            self.secondary_range.name: [],
            self.input_title: [],
        }
        self.input_data = []
        for secondary_value in self.secondary_range.data_values:
            series_name = f"${self.secondary_range.name}={secondary_value:.2f}$"

            print(f"Evaluating {self.staging_type.name} ({series_name})")
            try:
                inputs, coefficients = get_inverse_values(
                    self.rating_range,
                    lambda x, target: self.calculation(
                        float(x), target, self.staging_type, secondary_value
                    ),
                    self.initial_guess,
                    curve_fit_function=self.curve_fit.function,
                    curve_fit_guesses=self.curve_fit.initial_coefficient_guesses,
                    do_curve_fit=do_curve_fit,
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
                    line_properties=MarkersOnly() if do_curve_fit else LineProperties(),
                    y_axis_name=self.input_title,
                )
            )
            if do_curve_fit:
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
                        y_axis_name=self.input_title,
                    )
                )

            csv_output_data[self.rating_range.name] += list(
                self.rating_range.data_values
            )
            csv_output_data[self.secondary_range.name] += [secondary_value] * len(
                self.rating_range.data_values
            )
            csv_output_data[self.input_title] += inputs
            self.input_data.append([float(v) for v in inputs])

        self.plot(display_data, output_name)

        self.write_csv(csv_output_data, output_name)

    def plot(self, display_data_list, figure_name):
        plot = DimensionalPlot(self.rating_range)  # , title=self.staging_type.name)
        for y in display_data_list:
            plot.add_display_data(y)
        plot.write_html_plot(f"output/{figure_name}.html")
        plot.write_image_plot(f"output/{figure_name}.png")
        plot.write_image_plot(f"output/{figure_name}.pdf")
        time.sleep(2)
        plot.write_image_plot(f"output/{figure_name}.pdf")

    def write_csv(self, column_dictionary: Dict[str, List[float]], table_name: str):
        with open(f"output/{table_name}-points.csv", "w", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            key_list = list(column_dictionary.keys())
            writer.writerow(key_list)
            for index in range(len(column_dictionary[key_list[0]])):
                writer.writerow([column_dictionary[key][index] for key in key_list])

    def write_csv2(self, table_name: str, write_mode: str = "w"):
        with open(
            f"output/{table_name}-table.csv", write_mode, encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [f"{self.staging_type.name} {self.input_title}", self.rating_range.name]
            )
            writer.writerow(
                [self.secondary_range.name]
                + [float(v) for v in list(self.rating_range.data_values)]
            )
            for index, rating_value in enumerate(self.secondary_range.data_values):
                writer.writerow([rating_value] + self.input_data[index])
            writer.writerow(["", ""])


def make_objective_function(comparison, target):
    return lambda x: comparison(x, target) - target


def get_inverse_values(
    target_range,
    metric_calculation,
    initial_guess=lambda x: x / 3.0,
    curve_fit_function=quadratic,
    curve_fit_guesses=(1, 1, 1),
    do_curve_fit=False,
):
    inverse_values = []
    for target in target_range.data_values:
        print(f"    {target_range.name}={target:.1f}")
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
    if do_curve_fit:
        print(f"    curve fitting...")
        curve_fit_coefficients = optimize.curve_fit(
            curve_fit_function,
            target_range.data_values,
            inverse_values,
            curve_fit_guesses,
        )[0]
    else:
        curve_fit_coefficients = None
    return inverse_values, curve_fit_coefficients


# Cooling
def seer_function(cop_82_min, target_seer, staging_type, seer_eer_ratio):
    eer = target_seer / seer_eer_ratio
    seer = resdx.DXUnit(
        staging_type=staging_type,
        input_seer=target_seer,
        input_eer=eer,
        rated_net_total_cooling_cop_82_min=cop_82_min,
        input_hspf=7.5,
    ).seer()
    return seer


two_speed_cooling_regression = RatingRegression(
    staging_type=resdx.StagingType.TWO_STAGE,
    calculation=seer_function,
    input_title="$COP_{82°F,low}$",
    initial_guess=lambda target: target / 3.0,
    rating_range=DimensionalData(
        [float(v) for v in list(linspace(6, 22, 2))],
        name="$SEER2$",
        native_units="Btu/Wh",
    ),  # All straight lines don't need more than two points
    secondary_range=DimensionalData(
        [float(v) for v in list(linspace(1.0, 2.4, 2))],
        name="SEER2/EER2",
        native_units="W/W",
    ),  # Also straight line
    curve_fit=linear_curve_fit,
)

variable_speed_cooling_regression = deepcopy(two_speed_cooling_regression)
variable_speed_cooling_regression.staging_type = resdx.StagingType.VARIABLE_SPEED
variable_speed_cooling_regression.rating_range = DimensionalData(
    [float(v) for v in list(linspace(14, 35, 3))], name="$SEER2$", native_units="Btu/Wh"
)  # Slight inflection, three points should suffice
variable_speed_cooling_regression.secondary_range = DimensionalData(
    [float(v) for v in list(geometric_space(1.0, 2.4, 5, 0.5))],
    name="SEER2/EER2",
    native_units="W/W",
)  # Exponential variation 5 values


# Heating
def hspf_function(cop_47, hspf, staging_type, cap17m):
    # Note: results are sensitive to cap95 since it is used to set the building load and external static pressure.
    cap95 = fr_u(3.0, "ton_ref")
    cap47 = (
        cap95
        if staging_type != resdx.StagingType.VARIABLE_SPEED
        else resdx.models.tabular_data.neep_cap47_from_cap95(cap95)
    )
    cap17 = cap47 * cap17m
    return resdx.DXUnit(
        staging_type=staging_type,
        rated_net_total_cooling_capacity=cap95,
        rated_net_heating_capacity=cap47,
        rated_net_heating_capacity_17=cap17,
        rated_net_heating_cop=cop_47,
        input_hspf=hspf,
        input_seer=14.3,
        input_eer=11.0,
    ).hspf()


single_speed_heating_regression = RatingRegression(
    staging_type=resdx.StagingType.SINGLE_STAGE,
    calculation=hspf_function,
    input_title="$COP_{47°F,full}$",
    initial_guess=lambda target: target / 2.0,
    rating_range=DimensionalData(
        [float(v) for v in list(linspace(5, 11, 5))],
        name="$HSPF2$",
        native_units="Btu/Wh",
    ),
    secondary_range=DimensionalData(
        [float(v) for v in list(geometric_space(0.5, 1.0, 5, 2.0))],
        name="Q_{17°F,full}/Q_{47°F,full}",
        native_units="Btu/Btu",
    ),
    curve_fit=quadratic_curve_fit,
)

two_speed_heating_regression = deepcopy(single_speed_heating_regression)
two_speed_heating_regression.staging_type = resdx.StagingType.TWO_STAGE

variable_speed_heating_regression = deepcopy(single_speed_heating_regression)
variable_speed_heating_regression.staging_type = resdx.StagingType.VARIABLE_SPEED
variable_speed_heating_regression.rating_range = DimensionalData(
    [float(v) for v in list(linspace(7, 16, 5))], name="$HSPF2$", native_units="Btu/Wh"
)
variable_speed_heating_regression.secondary_range = DimensionalData(
    [float(v) for v in list(geometric_space(0.5, 1.1, 5, 2.0))],
    name="Q_{17°F,full}/Q_{47°F,full}",
    native_units="Btu/Btu",
)

two_speed_cooling_regression.evaluate("seer-two")
variable_speed_cooling_regression.evaluate("seer-variable")
single_speed_heating_regression.evaluate("hspf-one")
two_speed_heating_regression.evaluate("hspf-two")
variable_speed_heating_regression.evaluate("hspf-variable")

two_speed_cooling_regression.write_csv2("regressions")
variable_speed_cooling_regression.write_csv2("regressions", "a")
single_speed_heating_regression.write_csv2("regressions", "a")
two_speed_heating_regression.write_csv2("regressions", "a")
variable_speed_heating_regression.write_csv2("regressions", "a")

# Write python file
SOURCE_PATH = Path("resdx", "models")
FILE_NAME = "rating_correlations.py"
template_environment = Environment(loader=FileSystemLoader(SOURCE_PATH))
template = template_environment.get_template(f"{FILE_NAME}.jinja")

content = template.render(
    heating_1s=single_speed_heating_regression,
    heating_2s=two_speed_heating_regression,
    heating_vs=variable_speed_heating_regression,
    cooling_2s=two_speed_cooling_regression,
    cooling_vs=variable_speed_cooling_regression,
)

with open(Path(SOURCE_PATH, FILE_NAME), mode="w", encoding="utf-8") as python_file:
    python_file.write(content)

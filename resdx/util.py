import numpy as np


def limit_check(value, min=None, max=None):
    if min is not None:
        if value < min:
            raise Exception(f"Value, {value}, is less than the allowed minimum of {min}")

    if max is not None:
        if value > max:
            raise Exception(f"Value, {value}, is greater than the allowed maximum of {max}")

    return value


def bracket(value, min=None, max=None):
    if min is not None:
        if value < min:
            return min

    if max is not None:
        if value > max:
            return max

    return value


def set_default(input, default, number_of_speeds=1):
    if input is None:
        return default
    else:
        if isinstance(default, list):
            if isinstance(input, list):
                return input
            else:
                return [input] * number_of_speeds
        else:
            return input


def make_list(input) -> list:
    if isinstance(input, list):
        return input
    else:
        return [input]


def calc_biquad(coeff, in_1, in_2):
    return (
        coeff[0]
        + coeff[1] * in_1
        + coeff[2] * in_1 * in_1
        + coeff[3] * in_2
        + coeff[4] * in_2 * in_2
        + coeff[5] * in_1 * in_2
    )


def calc_quad(coeff, in_1):
    return coeff[0] + coeff[1] * in_1 + coeff[2] * in_1 * in_1


def find_nearest(array, value):
    closest_diff = abs(array[0] - value)
    closest_value = array[0]
    for option in array:
        diff = abs(option - value)
        if diff < closest_diff:
            closest_diff = diff
            closest_value = option
    return closest_value


# Used for curve-fitting. TODO: Use to replace the functions above?


def linear(x, c0, c1):
    return c0 + c1 * x


def linear_string(x_name, c0, c1):
    return f"{c0:.4g} + {c1:.4g}*{x_name}"


def quadratic(x, c0, c1, c2):
    return c0 + c1 * x + c2 * x * x


def quadratic_string(x_name, c0, c1, c2):
    return f"{c0:.4g} + {c1:.4g}*{x_name} + {c2:.4g}*{x_name}^2"


def cubic(x, c0, c1, c2, c3):
    return c0 + c1 * x + c2 * x * x + c3 * x * x * x


def cubic_string(x_name, c0, c1, c2, c3):
    return f"{c0:.4g} + {c1:.4g}*{x_name} + {c2:.4g}*{x_name}^2 + {c3:.4g}*{x_name}^3"


def quartic(x, c0, c1, c2, c3, c4):
    x2 = x * x
    return c0 + c1 * x + c2 * x2 + c3 * x2 * x + c4 * x2 * x2


def quartic_string(x_name, c0, c1, c2, c3, c4):
    return f"{c0:.4g} + {c1:.4g}*{x_name} + {c2:.4g}*{x_name}^2 + {c3:.4g}*{x_name}^3 + {c4:.4g}*{x_name}^4"


def exponential(x, c0, c1, c2):
    return c0 + c1 ** (c2 * x)


def exponential_string(x_name, c0, c1, c2):
    return f"{c0:.4g} + {c1:.4g}^({c2:.4g}*{x_name})"


def calculate_r_squared(source_data, regression_data):
    source_data_array = np.array(source_data)
    residuals = source_data_array - np.array(regression_data)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((source_data_array - np.mean(source_data_array)) ** 2)
    return 1 - ss_res / ss_tot


def geometric_space(start: float, end: float, number: int, coefficient: float = 1.0) -> list[float]:
    distance = end - start
    if coefficient == 1.0:
        d0 = distance / (number - 1)
    else:
        d0 = distance * (coefficient - 1) / (coefficient ** (number - 1) - 1)
    values = np.zeros(number)
    values[0] = start
    for index in range(len(values) - 1):
        delta = d0 * (coefficient**index)
        values[index + 1] = values[index] + delta
    return [float(value) for value in values]

from copy import deepcopy

from koozie import fr_u
from pytest import approx, mark, raises

from resdx import RatingStandard, StagingType
from resdx.performance_points import PerformancePoint, PowerLimit, apply_performance_points, apply_power_limits
from resdx.rating_solver import make_rating_unit

# NEEP statistical low speed capacity is below the Appendix F range
pytestmark = mark.filterwarnings("ignore:Low speed capacity at 82")

ROOM_STANDARD = RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024
FULL_LOAD_SPEEDS = {"cooling": 2, "heating": 2}  # Variable speed table speed numbers
T_MINUS_20 = fr_u(-20.0, "°F")  # Heating off temperature
T_5 = fr_u(5.0, "°F")
T_17 = fr_u(17.0, "°F")
T_30 = fr_u(30.0, "°F")
T_47 = fr_u(47.0, "°F")
T_62 = fr_u(62.0, "°F")
T_95 = fr_u(95.0, "°F")


def make_room_unit(**kwargs):
    arguments = {
        "staging_type": StagingType.VARIABLE_SPEED,
        "eer": 11.26,
        "ceer": 16.8,
        "heer": 10.9,
        "q95": fr_u(9000, "Btu/h"),
        "t_min": T_MINUS_20,
        "rating_standard": ROOM_STANDARD,
    }
    arguments.update(kwargs)
    return make_rating_unit(**arguments)


def base_map():
    return deepcopy(make_room_unit().net_tabular_data)


def test_pin_capacity_and_cop():
    performance_map = base_map()
    original = deepcopy(performance_map)
    capacity = fr_u(7200, "Btu/h")
    apply_performance_points(
        performance_map, [PerformancePoint("heating", 2, T_17, capacity=capacity, cop=1.78)], FULL_LOAD_SPEEDS
    )

    capacities, powers = performance_map.heating_capacities, performance_map.heating_powers
    assert capacities.get(2, T_17) == approx(capacity)
    assert capacities.get(2, T_17) / powers.get(2, T_17) == approx(1.78)
    # Slower speeds are unchanged
    assert capacities.get(1, T_17) == original.heating_capacities.get(1, T_17)
    # Beyond the warmest and coldest pin, the speed is scaled to stay continuous with the pin
    factor = capacity / original.heating_capacities.get(2, T_17)
    assert capacities.get(2, T_47) == approx(original.heating_capacities.get(2, T_47) * factor)
    assert capacities.get(2, T_5) == approx(original.heating_capacities.get(2, T_5) * factor)
    # Boost speeds keep their ratio to the pinned full load speed
    assert capacities.get(3, T_17) / capacities.get(2, T_17) == approx(
        original.heating_capacities.get(3, T_17) / original.heating_capacities.get(2, T_17)
    )


def test_pin_minimum_speed_only():
    # Pinning a slower speed (e.g., for turndown) leaves full load and boost unchanged
    performance_map = base_map()
    original = deepcopy(performance_map)
    apply_performance_points(
        performance_map, [PerformancePoint("cooling", 1, T_95, capacity=fr_u(4500, "Btu/h"))], FULL_LOAD_SPEEDS
    )
    capacities = performance_map.cooling_capacities
    assert capacities.get(1, T_95) == approx(fr_u(4500, "Btu/h"))
    for speed in (2, 3):
        for t in capacities.temperatures:
            assert capacities.get(speed, t) == original.cooling_capacities.get(speed, t)


def test_pins_interpolate_between_and_scale_beyond():
    performance_map = base_map()
    original = deepcopy(performance_map)
    points = [
        PerformancePoint("heating", 2, T_47, capacity=fr_u(9000, "Btu/h"), cop=3.2),
        PerformancePoint("heating", 2, T_30, capacity=fr_u(8000, "Btu/h"), cop=2.6),  # New temperature
        PerformancePoint("heating", 2, T_5, capacity=fr_u(4500, "Btu/h"), cop=1.25),
    ]
    apply_performance_points(performance_map, points, FULL_LOAD_SPEEDS)

    capacities = performance_map.heating_capacities
    assert T_30 in capacities.temperatures
    assert capacities.get(2, T_30) == approx(fr_u(8000, "Btu/h"))
    # Unpinned temperatures between pins keep their values
    assert capacities.get(2, T_17) == approx(original.heating_capacities.get(2, T_17))
    # Colder than the coldest pin, scaled by the coldest pin's factor
    factor_5 = fr_u(4500, "Btu/h") / original.heating_capacities.get(2, T_5)
    assert capacities.get(2, T_MINUS_20) == approx(original.heating_capacities.get(2, T_MINUS_20) * factor_5)


def test_single_quantity_pins():
    original = base_map()
    original_cop = original.heating_capacities.get(2, T_17) / original.heating_powers.get(2, T_17)

    performance_map = deepcopy(original)
    apply_performance_points(
        performance_map, [PerformancePoint("heating", 2, T_17, capacity=fr_u(8000, "Btu/h"))], FULL_LOAD_SPEEDS
    )
    capacities, powers = performance_map.heating_capacities, performance_map.heating_powers
    assert capacities.get(2, T_17) / powers.get(2, T_17) == approx(original_cop)  # Capacity alone keeps COP

    performance_map = deepcopy(original)
    apply_performance_points(performance_map, [PerformancePoint("heating", 2, T_17, cop=2.0)], FULL_LOAD_SPEEDS)
    capacities, powers = performance_map.heating_capacities, performance_map.heating_powers
    assert capacities.get(2, T_17) == approx(original.heating_capacities.get(2, T_17))  # COP alone keeps capacity
    assert capacities.get(2, T_17) / powers.get(2, T_17) == approx(2.0)

    performance_map = deepcopy(original)
    apply_performance_points(performance_map, [PerformancePoint("heating", 2, T_17, power=1000.0)], FULL_LOAD_SPEEDS)
    capacities, powers = performance_map.heating_capacities, performance_map.heating_powers
    assert powers.get(2, T_17) == approx(1000.0)
    assert capacities.get(2, T_17) / powers.get(2, T_17) == approx(original_cop)  # Power alone keeps COP


def test_invalid_pins():
    performance_map = base_map()
    with raises(ValueError, match="one or two of capacity, cop, and power"):
        apply_performance_points(performance_map, [PerformancePoint("heating", 2, T_17)], FULL_LOAD_SPEEDS)
    with raises(ValueError, match="inconsistent"):
        apply_performance_points(
            deepcopy(performance_map),
            [PerformancePoint("heating", 2, T_17, capacity=fr_u(8000, "Btu/h"), cop=2.0, power=10.0)],
            FULL_LOAD_SPEEDS,
        )
    with raises(ValueError, match="out of speed order"):
        apply_performance_points(
            deepcopy(performance_map), [PerformancePoint("heating", 2, T_17, capacity=1.0)], FULL_LOAD_SPEEDS
        )


def test_power_limit():
    performance_map = base_map()
    original = deepcopy(performance_map)
    maximum_power = original.heating_powers.get(2, T_5) * 0.9
    apply_power_limits(performance_map, [PowerLimit("heating", maximum_power, T_MINUS_20, T_62)])

    capacities, powers = performance_map.heating_capacities, performance_map.heating_powers
    # The range endpoints are added so the limit holds across the range
    assert T_62 in powers.temperatures
    for t in powers.temperatures:
        for speed in powers.speeds:
            assert powers.get(speed, t) <= maximum_power * (1.0 + 1.0e-9)
    # The limited speed moves down the speed curve: capacity drops and COP moves toward the lower speed's COP
    assert powers.get(2, T_5) == approx(maximum_power)
    assert capacities.get(2, T_5) < original.heating_capacities.get(2, T_5)
    original_cop = original.heating_capacities.get(2, T_5) / original.heating_powers.get(2, T_5)
    assert capacities.get(2, T_5) / powers.get(2, T_5) > original_cop

    with raises(ValueError, match="Minimum speed power"):
        apply_power_limits(deepcopy(original), [PowerLimit("heating", 1.0, T_MINUS_20, T_62)])


def test_rating_unit_with_pins_and_limits():
    capacity_17 = fr_u(7200, "Btu/h")
    unit = make_room_unit(
        performance_points=[
            PerformancePoint("heating", 2, T_17, capacity=capacity_17),
            PerformancePoint("cooling", 1, T_95, capacity=fr_u(4500, "Btu/h")),
        ],
        power_limits=[PowerLimit("heating", 1200.0, T_MINUS_20, T_62)],
    )
    capacities, powers = unit.net_tabular_data.heating_capacities, unit.net_tabular_data.heating_powers
    assert capacities.get(2, T_17) == approx(capacity_17)
    assert unit.net_tabular_data.cooling_capacities.get(1, T_95) == approx(fr_u(4500, "Btu/h"))
    assert max(max(row) for row in powers.data) <= 1200.0 * (1.0 + 1.0e-9)
    assert unit.ceer() == approx(16.8, 0.0001)
    assert unit.heer() == approx(10.9, 0.0001)


def test_rating_unit_adjustment_point():
    # HEER is matched by adjusting the full speed COP at 17 °F while the 47 °F COP stays pinned
    unit = make_room_unit(
        heer=9.0,
        performance_points=[PerformancePoint("heating", 2, T_47, capacity=fr_u(9000, "Btu/h"), cop=3.2)],
        heating_adjustment_point=PerformancePoint("heating", 2, T_17),
    )
    data = unit.net_tabular_data
    assert data.heating_capacities.get(2, T_47) / data.heating_powers.get(2, T_47) == approx(3.2)
    assert unit.heer() == approx(9.0, 0.0001)

    with raises(ValueError, match="pinned"):
        make_room_unit(
            performance_points=[PerformancePoint("heating", 2, T_47, cop=3.2)],
        )


def test_rating_unit_without_heer():
    # Without a HEER target, the pinned COP at the default adjustment point (full speed, 47 °F) is the model input
    unit = make_room_unit(
        heer=None,
        performance_points=[PerformancePoint("heating", 2, T_47, capacity=fr_u(9000, "Btu/h"), cop=3.2)],
    )
    data = unit.net_tabular_data
    assert data.heating_capacities.get(2, T_47) / data.heating_powers.get(2, T_47) == approx(3.2)
    assert unit.ceer() == approx(16.8, 0.0001)

    with raises(ValueError, match="heer or a pinned heating COP"):
        make_room_unit(heer=None)

"""Pinned performance points and power limits for performance maps."""

from dataclasses import dataclass

from koozie import to_u

from .models.tabular_data import TemperatureSpeedPerformance, TemperatureSpeedPerformanceTable

TEMPERATURE_TOLERANCE = 1.0e-6  # K
CONSISTENCY_TOLERANCE = 1.0e-6  # Relative


@dataclass
class PerformancePoint:
    """A point in a performance map, with one or two of capacity, COP, and power.

    mode: "cooling" or "heating"
    speed: Performance table speed number (1 is the lowest speed)
    temperature: Outdoor drybulb temperature (K)
    capacity: Net capacity (W). Given alone, the point's COP is kept.
    cop: Net COP. Given alone, the point's capacity is kept.
    power: Net power (W). Given alone, the point's COP is kept.

    As a rating adjustment point (see make_rating_unit), capacity, cop, and power are not given.
    """

    mode: str
    speed: int
    temperature: float
    capacity: float | None = None
    cop: float | None = None
    power: float | None = None

    def is_at(self, mode: str, speed: int, temperature: float) -> bool:
        return self.mode == mode and self.speed == speed and abs(self.temperature - temperature) < TEMPERATURE_TOLERANCE

    def pinned_values(self, capacity: float, power: float) -> tuple[float, float]:
        """Capacity and power (W) at the point, given the map's current capacity and power there."""
        values = (self.capacity, self.cop, self.power)
        if all(value is None for value in values):
            raise ValueError(f"Performance point requires one or two of capacity, cop, and power: {self}")
        if self.capacity is not None and self.power is not None:
            if self.cop is not None and abs(self.capacity / self.power - self.cop) > CONSISTENCY_TOLERANCE * self.cop:
                raise ValueError(f"Performance point capacity, cop, and power are inconsistent: {self}")
            return self.capacity, self.power
        cop = self.cop if self.cop is not None else capacity / power  # Capacity or power alone keeps the COP
        if self.capacity is not None:
            return self.capacity, self.capacity / cop
        if self.power is not None:
            return self.power * cop, self.power
        return capacity, capacity / cop  # COP alone keeps the capacity


@dataclass
class PowerLimit:
    """Maximum net power (W) for a mode, guaranteed between minimum_temperature and maximum_temperature (K)."""

    mode: str
    maximum_power: float
    minimum_temperature: float
    maximum_temperature: float


def get_tables(
    performance_map: TemperatureSpeedPerformance, mode: str
) -> tuple[TemperatureSpeedPerformanceTable, TemperatureSpeedPerformanceTable]:
    """Capacity and power tables for a mode ("cooling" or "heating")."""
    if mode == "cooling":
        return performance_map.cooling_capacities, performance_map.cooling_powers
    if mode == "heating":
        return performance_map.heating_capacities, performance_map.heating_powers
    raise ValueError(f'Unknown mode, "{mode}".')


def table_temperature(table: TemperatureSpeedPerformanceTable, temperature: float) -> float:
    """The table's temperature matching the given temperature, adding it to the table if needed."""
    for existing in table.temperatures:
        if abs(existing - temperature) < TEMPERATURE_TOLERANCE:
            return existing
    table.add_temperature(temperature)
    return temperature


def pin_speed(
    capacities: TemperatureSpeedPerformanceTable,
    powers: TemperatureSpeedPerformanceTable,
    speed: int,
    points: list[PerformancePoint],
    faster_speeds: list[int],
) -> None:
    """Pin a speed's points, scale the speed beyond its warmest and coldest pins, and scale the faster speeds to keep
    their ratio to it."""
    original_capacities = {t: capacities.get(speed, t) for t in capacities.temperatures}
    original_powers = {t: powers.get(speed, t) for t in powers.temperatures}
    pins: dict[float, tuple[float, float]] = {}
    for point in points:
        t = table_temperature(capacities, point.temperature)
        pins[t] = point.pinned_values(original_capacities[t], original_powers[t])

    coldest, warmest = min(pins), max(pins)
    for t in capacities.temperatures:
        if t in pins:
            capacity, power = pins[t]
        elif coldest < t < warmest:
            continue
        else:
            edge = coldest if t < coldest else warmest
            capacity = original_capacities[t] * pins[edge][0] / original_capacities[edge]
            power = original_powers[t] * pins[edge][1] / original_powers[edge]
        capacities.set(speed, t, capacity)
        powers.set(speed, t, power)

    for faster_speed in faster_speeds:
        for t in capacities.temperatures:
            capacity_ratio = capacities.get(speed, t) / original_capacities[t]
            power_ratio = powers.get(speed, t) / original_powers[t]
            capacities.set(faster_speed, t, capacities.get(faster_speed, t) * capacity_ratio)
            powers.set(faster_speed, t, powers.get(faster_speed, t) * power_ratio)


def apply_performance_points(performance_map: TemperatureSpeedPerformance, points: list[PerformancePoint]) -> None:
    """Pin performance points in the map.

    Each point becomes a table point (added if needed). Unpinned temperatures between a speed's pins keep their values,
    and temperatures beyond its warmest and coldest pins are scaled to stay continuous with them. Faster speeds without
    pins keep their ratio to the next slower pinned speed.
    """
    for mode in ("cooling", "heating"):
        mode_points = [point for point in points if point.mode == mode]
        if not mode_points:
            continue
        capacities, powers = get_tables(performance_map, mode)
        for point in mode_points:
            table_temperature(capacities, point.temperature)
            table_temperature(powers, point.temperature)

        pinned_speeds = sorted({point.speed for point in mode_points})
        for i, speed in enumerate(pinned_speeds):
            next_pinned_speed = pinned_speeds[i + 1] if i + 1 < len(pinned_speeds) else capacities.speeds[-1] + 1
            faster_speeds = [s for s in capacities.speeds if speed < s < next_pinned_speed]
            speed_points = [point for point in mode_points if point.speed == speed]
            pin_speed(capacities, powers, speed, speed_points, faster_speeds)

        for t in capacities.temperatures:
            speed_capacities = [capacities.get(speed, t) for speed in capacities.speeds]
            if any(lower > higher for lower, higher in zip(speed_capacities, speed_capacities[1:])):
                raise ValueError(f"Pinned {mode} capacities are out of speed order at {to_u(t, '°F'):.1f} °F.")


def limit_power(
    capacities: TemperatureSpeedPerformanceTable, powers: TemperatureSpeedPerformanceTable, maximum_power: float
) -> None:
    """Move speeds whose power exceeds the maximum down the speed curve to where power equals the maximum.

    Like a compressor held to a lower speed, the limited speed's capacity drops and its COP moves toward the lower
    speed's COP. Faster speeds get the same performance.
    """
    for t in powers.temperatures:
        if powers.get(1, t) > maximum_power:
            raise ValueError(
                f"Minimum speed power ({powers.get(1, t):.0f} W) at {to_u(t, '°F'):.1f} °F exceeds the maximum power "
                f"({maximum_power:.0f} W)."
            )
        for speed in powers.speeds[1:]:
            lower_power = powers.get(speed - 1, t)
            power = powers.get(speed, t)
            if power <= maximum_power:
                continue
            fraction = (maximum_power - lower_power) / (power - lower_power)
            lower_capacity = capacities.get(speed - 1, t)
            capacity = lower_capacity + fraction * (capacities.get(speed, t) - lower_capacity)
            for limited_speed in powers.speeds[speed - 1 :]:
                capacities.set(limited_speed, t, capacity)
                powers.set(limited_speed, t, maximum_power)
            break


def apply_power_limits(performance_map: TemperatureSpeedPerformance, limits: list[PowerLimit]) -> None:
    """Limit power in the map. Each limit's temperature range endpoints are added to the table, so the limit holds
    between them (the map is linear between table temperatures)."""
    for limit in limits:
        capacities, powers = get_tables(performance_map, limit.mode)
        for temperature in (limit.minimum_temperature, limit.maximum_temperature):
            table_temperature(capacities, temperature)
            table_temperature(powers, temperature)
        limit_power(capacities, powers, limit.maximum_power)

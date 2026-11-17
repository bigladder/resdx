from collections.abc import Callable
from copy import deepcopy

from koozie import fr_u, to_u
from scipy import optimize

from .dx_unit import RatingStandard, StagingType
from .models.tabular_data import TemperatureSpeedPerformance
from .models.unified_resnet import RESNETDXModel
from .performance_points import PerformancePoint, PowerLimit, apply_performance_points, apply_power_limits


def pinned_cop(points: list[PerformancePoint], location: PerformancePoint | None) -> float | None:
    """COP set by a pinned point at the location, if any."""
    if location is None:
        return None
    for point in points:
        if not point.is_at(location.mode, location.speed, location.temperature):
            continue
        if point.cop is not None:
            return point.cop
        if point.capacity is not None and point.power is not None:
            return point.capacity / point.power
    return None


def sets_cop(points: list[PerformancePoint], location: PerformancePoint) -> bool:
    """Whether a pinned point sets the COP at the location (a capacity alone keeps the existing COP)."""
    return any(
        point.is_at(location.mode, location.speed, location.temperature)
        and (point.cop is not None or point.power is not None)
        for point in points
    )


def with_adjustment(points: list[PerformancePoint], location: PerformancePoint, cop: float) -> list[PerformancePoint]:
    """Points with the COP at the adjustment location set (keeping a capacity pinned there)."""
    at_location = [point for point in points if point.is_at(location.mode, location.speed, location.temperature)]
    capacity = next((point.capacity for point in at_location if point.capacity is not None), None)
    adjusted = [point for point in points if point not in at_location]
    adjusted.append(PerformancePoint(location.mode, location.speed, location.temperature, capacity=capacity, cop=cop))
    return adjusted


def make_rating_unit(
    staging_type: StagingType,
    seer: float | None = None,
    hspf: float | None = None,
    eer: float | None = None,
    q95: float = fr_u(3.0, "ton_ref"),
    q47: float | None = None,
    qm17: float = 0.63,
    t_min: float | None = None,
    t_defrost: float = fr_u(40.0, "degF"),
    rating_standard: RatingStandard = RatingStandard.AHRI_210_240_2023,
    ceer: float | None = None,
    heer: float | None = None,
    inactive_mode_power: float = 0.0,
    off_mode_power: float = 0.0,
    modify_map: Callable[[TemperatureSpeedPerformance], None] | None = None,
    performance_points: list[PerformancePoint] | None = None,
    power_limits: list[PowerLimit] | None = None,
    cooling_adjustment_point: PerformancePoint | None = None,
    heating_adjustment_point: PerformancePoint | None = None,
) -> RESNETDXModel:
    """Make a unit that matches the given ratings: SEER and HSPF for AHRI 210/240 rating standards, or CEER and HEER
    for room units (RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024).

    The ratings are matched with these modifications to the net performance map applied, in this order:
    modify_map: Modifies the map in place.
    performance_points: Pinned points (see apply_performance_points).
    power_limits: Power limits (see apply_power_limits).

    By default, the solver matches the cooling rating by adjusting the low speed COP at 82 °F (the full speed COP at
    95 °F for single speed room units) and the heating rating by adjusting the full speed COP at 47 °F. A
    cooling_adjustment_point or heating_adjustment_point (a PerformancePoint without values) moves the adjustment to
    another point, whose COP the solver sets at its existing capacity. An adjustment point cannot have a pinned COP.

    For room units, ceer or heer may be omitted if the default adjustment point has a pinned COP. That COP is then used
    to make the map, and the rating is a result."""
    points = list(performance_points or [])
    limits = list(power_limits or [])
    is_room_unit = rating_standard == RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024

    if is_room_unit:
        if seer is not None or hspf is not None:
            raise ValueError(f"seer and hspf are not used for {rating_standard}.")
        if staging_type == StagingType.SINGLE_STAGE:
            if eer is not None:
                raise ValueError("eer is determined from ceer for single speed room units.")
        elif eer is None:
            raise ValueError(f"eer is required for {rating_standard}.")
    else:
        if seer is None or hspf is None:
            raise ValueError(f"seer and hspf are required for {rating_standard}.")
        if ceer is not None or heer is not None:
            raise ValueError(f"ceer and heer are not used for {rating_standard}.")

    q47_: float
    if q47 is None:
        q47_ = q95
    else:
        q47_ = q47

    def make_base_unit(cop_82_min: float, cop_47: float | None, eer_: float | None) -> RESNETDXModel:
        """Unit without map modifications."""
        return RESNETDXModel(
            staging_type=staging_type,
            rating_standard=rating_standard,
            rated_net_total_cooling_capacity=q95,
            rated_net_heating_capacity=q47,
            rated_net_heating_capacity_17=q47_ * qm17,
            input_seer=seer,
            input_eer=eer_,
            input_hspf=hspf,
            rated_net_heating_cop=cop_47,
            rated_net_total_cooling_cop_82_min=cop_82_min,
            heating_off_temperature=t_min,
            defrost_temperature=t_defrost,
            is_ducted=not is_room_unit,
            inactive_mode_power=inactive_mode_power,
            off_mode_power=off_mode_power,
        )

    # Table speed numbers of the full load and low speeds (independent of the arbitrary COPs used here)
    speed_unit = make_base_unit(
        3.0, 3.0 if is_room_unit else None, eer if eer is not None or not is_room_unit else 10.0
    )
    full_load_speeds = {
        "cooling": round(speed_unit.tabular_cooling_speed_map[speed_unit.cooling_full_load_speed]),
        "heating": round(speed_unit.tabular_heating_speed_map[speed_unit.heating_full_load_speed]),
    }

    default_cooling_location: PerformancePoint | None
    if is_room_unit and staging_type == StagingType.SINGLE_STAGE:
        default_cooling_location = PerformancePoint("cooling", full_load_speeds["cooling"], fr_u(95.0, "°F"))
    elif staging_type != StagingType.SINGLE_STAGE:
        low_speed = round(speed_unit.tabular_cooling_speed_map[speed_unit.cooling_low_speed])
        default_cooling_location = PerformancePoint("cooling", low_speed, fr_u(82.0, "°F"))
    else:
        default_cooling_location = None  # SEER is a model input
    default_heating_location = PerformancePoint("heating", full_load_speeds["heating"], fr_u(47.0, "°F"))

    if is_room_unit:
        if ceer is None and pinned_cop(points, default_cooling_location) is None:
            raise ValueError(
                f"ceer or a pinned cooling COP at the default adjustment point is required for {rating_standard}."
            )
        if heer is None and pinned_cop(points, default_heating_location) is None:
            raise ValueError(
                f"heer or a pinned heating COP at the default adjustment point (full speed, 47 °F) is required for "
                f"{rating_standard}."
            )

    cooling_target = ceer if is_room_unit else seer
    heating_target = heer if is_room_unit else hspf
    for target, adjustment_point, default_location in (
        (cooling_target, cooling_adjustment_point, default_cooling_location),
        (heating_target, heating_adjustment_point, default_heating_location),
    ):
        if adjustment_point is not None:
            if target is None:
                raise ValueError(f"An adjustment point requires a rating target: {adjustment_point}")
            if sets_cop(points, adjustment_point):
                raise ValueError(f"The adjustment point is pinned: {adjustment_point}")
        elif target is not None and default_location is not None and sets_cop(points, default_location):
            raise ValueError(
                f"The default adjustment point is pinned, so an adjustment point is required: {default_location}"
            )

    is_modified = modify_map is not None or bool(points) or bool(limits)

    def make_unit(
        cop_82_min: float,
        cop_47: float | None = None,
        eer_: float | None = eer,
        cooling_cop: float | None = None,
        heating_cop: float | None = None,
    ) -> RESNETDXModel:
        unit = make_base_unit(cop_82_min, cop_47, eer_)
        if not is_modified and cooling_cop is None and heating_cop is None:
            return unit
        performance_map = deepcopy(unit.net_tabular_data)
        if modify_map is not None:
            modify_map(performance_map)
        unit_points = points
        if cooling_cop is not None:
            unit_points = with_adjustment(unit_points, cooling_adjustment_point, cooling_cop)  # type: ignore[arg-type]
        if heating_cop is not None:
            unit_points = with_adjustment(unit_points, heating_adjustment_point, heating_cop)  # type: ignore[arg-type]
        if unit_points:
            apply_performance_points(performance_map, unit_points, full_load_speeds)
        if limits:
            apply_power_limits(performance_map, limits)
        performance_map.set_interpolators()
        return RESNETDXModel(
            staging_type=staging_type,
            rating_standard=rating_standard,
            tabular_data=performance_map,
            heating_off_temperature=t_min,
            defrost_temperature=t_defrost,
            is_ducted=not is_room_unit,
            inactive_mode_power=inactive_mode_power,
            off_mode_power=off_mode_power,
        )

    def cooling_rating(unit: RESNETDXModel) -> float:
        return unit.ceer() if is_room_unit else unit.seer()

    def heating_rating(unit: RESNETDXModel) -> float:
        return unit.heer() if is_room_unit else unit.hspf()

    def cop_at(unit: RESNETDXModel, location: PerformancePoint) -> float:
        data = unit.net_tabular_data
        if location.mode == "cooling":
            return float(data.cooling_cop(location.speed, location.temperature))
        return float(data.heating_cop(location.speed, location.temperature))

    # Without HSPF, a heating COP is needed to make the unit. Heating does not affect cooling ratings.
    cop_47_cooling: float | None = None
    if is_room_unit:
        cop_47_cooling = pinned_cop(points, default_heating_location) or 3.0

    # Cooling
    cop_82_min = 3.0  # Arbitrary value if it's not needed
    cooling_cop: float | None = None
    if cooling_target is None:
        cop = pinned_cop(points, default_cooling_location)
        if staging_type == StagingType.SINGLE_STAGE:
            eer = to_u(cop, "Btu/Wh")
        else:
            cop_82_min = cop  # type: ignore[assignment]
    elif cooling_adjustment_point is not None:
        cooling_cop = optimize.newton(
            lambda cop_guess: (
                cooling_rating(make_unit(cop_82_min, cop_47_cooling, eer, cooling_cop=cop_guess)) - cooling_target
            ),
            cop_at(make_unit(cop_82_min, cop_47_cooling, eer), cooling_adjustment_point),
        )
    elif is_room_unit:
        if staging_type == StagingType.SINGLE_STAGE:
            # Single speed CEER only depends on 95F performance
            eer = optimize.newton(
                lambda eer_guess: make_unit(cop_82_min, cop_47_cooling, eer_guess).ceer() - ceer,
                ceer,
            )
        else:
            cop_82_min = optimize.newton(
                lambda cop_82_min_guess: make_unit(cop_82_min_guess, cop_47_cooling).ceer() - ceer,
                ceer / 3.0,  # type: ignore[operator]
            )
    elif staging_type != StagingType.SINGLE_STAGE:
        cop_82_min = optimize.newton(
            lambda cop_82_min_guess: make_unit(cop_82_min_guess).seer() - seer,
            seer / 3.0,  # type: ignore[operator]
        )

    # Heating
    heating_cop: float | None = None
    cop_47: float | None
    if heating_target is None:
        cop_47 = pinned_cop(points, default_heating_location)
    elif heating_adjustment_point is not None:
        cop_47 = cop_47_cooling
        heating_cop = optimize.newton(
            lambda cop_guess: (
                heating_rating(make_unit(cop_82_min, cop_47, eer, cooling_cop=cooling_cop, heating_cop=cop_guess))
                - heating_target
            ),
            cop_at(make_unit(cop_82_min, cop_47, eer, cooling_cop=cooling_cop), heating_adjustment_point),
        )
    else:
        cop_47 = optimize.newton(
            lambda cop_47_guess: (
                heating_rating(make_unit(cop_82_min, cop_47_guess, eer, cooling_cop=cooling_cop)) - heating_target
            ),
            heating_target / 2.0,
        )

    return make_unit(cop_82_min, cop_47, eer, cooling_cop=cooling_cop, heating_cop=heating_cop)

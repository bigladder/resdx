from collections.abc import Callable
from copy import deepcopy

from koozie import fr_u
from scipy import optimize

from .dx_unit import RatingStandard, StagingType
from .models.tabular_data import TemperatureSpeedPerformance
from .models.unified_resnet import RESNETDXModel


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
) -> RESNETDXModel:
    """Make a unit that matches the given ratings: SEER and HSPF for AHRI 210/240 rating standards, or CEER and HEER
    for room units (RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024).

    modify_map, if provided, modifies the net performance map in place before the unit is rated, so the ratings are
    matched with the modifications applied."""
    is_room_unit = rating_standard == RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024
    if is_room_unit:
        if ceer is None or heer is None:
            raise ValueError(f"ceer and heer are required for {rating_standard}.")
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

    def make_unit(cop_82_min: float, cop_47: float | None = None, eer_: float | None = eer) -> RESNETDXModel:
        unit = RESNETDXModel(
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
        if modify_map is None:
            return unit
        performance_map = deepcopy(unit.net_tabular_data)
        modify_map(performance_map)
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

    # Without HSPF, a heating COP is needed to make the unit. Heating does not affect cooling ratings.
    cop_47_cooling = 3.0 if is_room_unit else None

    # Cooling COP 82 (B low)
    cop_82_min = 3.0  # Arbitrary value if it's not needed
    if is_room_unit:
        if staging_type == StagingType.SINGLE_STAGE:
            # Single speed CEER only depends on 95F performance
            eer = optimize.newton(
                lambda eer_guess: make_unit(cop_82_min, cop_47_cooling, eer_guess).ceer() - ceer,
                ceer,
            )
        else:
            cop_82_min = optimize.newton(
                lambda cop_82_min_guess: make_unit(cop_82_min_guess, cop_47_cooling).ceer() - ceer,
                ceer / 3.0,
            )
    elif staging_type != StagingType.SINGLE_STAGE:
        cop_82_min = optimize.newton(
            lambda cop_82_min_guess: make_unit(cop_82_min_guess).seer() - seer,
            seer / 3.0,
        )

    # Heating COP 47 (H1 Full)
    if is_room_unit:
        cop_47 = optimize.newton(
            lambda cop_47_guess: make_unit(cop_82_min, cop_47_guess, eer).heer() - heer,
            heer / 2.0,
        )
    else:
        cop_47 = optimize.newton(
            lambda cop_47_guess: make_unit(cop_82_min, cop_47_guess).hspf() - hspf,
            hspf / 2.0,
        )

    return make_unit(cop_82_min, cop_47, eer)

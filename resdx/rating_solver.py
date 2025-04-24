from koozie import fr_u
from scipy import optimize

from .dx_unit import AHRIVersion, StagingType
from .models.unified_resnet import RESNETDXModel


def make_rating_unit(
    staging_type: StagingType,
    seer: float,
    hspf: float,
    eer: float | None = None,
    q95: float = fr_u(3.0, "ton_ref"),
    q47: float | None = None,
    qm17: float = 0.63,
    t_min: float | None = None,
    t_defrost: float = fr_u(40.0, "degF"),
    rating_standard: AHRIVersion = AHRIVersion.AHRI_210_240_2023,
) -> RESNETDXModel:
    q47_: float
    if q47 is None:
        q47_ = q95
    else:
        q47_ = q47

    # Cooling COP 82 (B low)
    cop_82_min = 3.0  # Arbitrary value if it's not needed
    if staging_type != StagingType.SINGLE_STAGE:
        cop_82_min = optimize.newton(
            lambda cop_82_min_guess: RESNETDXModel(
                staging_type=staging_type,
                rating_standard=rating_standard,
                rated_net_total_cooling_capacity=q95,
                rated_net_heating_capacity=q47,
                rated_net_heating_capacity_17=q47_ * qm17,
                input_seer=seer,
                input_eer=eer,
                input_hspf=hspf,
                rated_net_total_cooling_cop_82_min=cop_82_min_guess,
                heating_off_temperature=t_min,
                defrost_temperature=t_defrost,
            ).seer()
            - seer,
            seer / 3.0,
        )

    # Heating COP 47 (H1 Full)
    cop_47 = optimize.newton(
        lambda cop_47_guess: RESNETDXModel(
            staging_type=staging_type,
            rating_standard=rating_standard,
            rated_net_total_cooling_capacity=q95,
            rated_net_heating_capacity=q47,
            rated_net_heating_capacity_17=q47_ * qm17,
            input_seer=seer,
            input_eer=eer,
            input_hspf=hspf,
            rated_net_heating_cop=cop_47_guess,
            rated_net_total_cooling_cop_82_min=cop_82_min,
            heating_off_temperature=t_min,
            defrost_temperature=t_defrost,
        ).hspf()
        - hspf,
        hspf / 2.0,
    )

    return RESNETDXModel(
        staging_type=staging_type,
        rating_standard=rating_standard,
        rated_net_total_cooling_capacity=q95,
        rated_net_heating_capacity=q47,
        rated_net_heating_capacity_17=q47_ * qm17,
        rated_net_heating_cop=cop_47,
        rated_net_total_cooling_cop_82_min=cop_82_min,
        input_seer=seer,
        input_eer=eer,
        input_hspf=hspf,
        heating_off_temperature=t_min,
        defrost_temperature=t_defrost,
    )

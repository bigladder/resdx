from pytest import approx

from scipy import optimize

from koozie import fr_u

from resdx import RESNETDXModel, StagingType, AHRIVersion, make_neep_model_data
from resdx.rating_solver import make_rating_unit


# Single speed gross COP values used for regression testing
COP_C = 3.312
COP_H = 3.020


# Tests
def test_1_speed_regression():
    # Single speed, SEER 13, HSPF 8
    seer = 13.0
    hspf = 8.0

    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        seer,
        hspf,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.gross_total_cooling_capacity() == approx(
        dx_unit_1_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_1_speed.cooling_fan_power() == approx(
        dx_unit_1_speed.rated_cooling_fan_power[0], 0.0001
    )
    assert dx_unit_1_speed.net_total_cooling_capacity() == approx(
        dx_unit_1_speed.rated_net_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_1_speed.seer() == approx(seer, 0.0001)
    assert dx_unit_1_speed.hspf() == approx(hspf, 0.0001)
    assert dx_unit_1_speed.rated_net_cooling_cop[0] == approx(COP_C, 0.001)
    assert dx_unit_1_speed.rated_net_heating_cop[0] == approx(COP_H, 0.001)


def test_1_speed_refrigerant_charge_regression():
    # Single speed, SEER 13, HSPF 8
    seer = 13.0
    hspf = 8.0

    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        seer,
        hspf,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )
    dx_unit_1_speed.refrigerant_charge_deviation = -0.25

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.seer() == approx(11.16, 0.01)
    assert dx_unit_1_speed.hspf() == approx(7.157, 0.01)


def test_1_speed_2023_regression():
    # Single speed, SEER 13, HSPF 8
    seer = 13.0
    hspf = 8.0

    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        seer,
        hspf,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )

    dx_unit_1_speed.set_rating_standard(AHRIVersion.AHRI_210_240_2023)

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.seer() == approx(13.000, 0.001)  # SEER2
    assert dx_unit_1_speed.hspf() == approx(7.108, 0.001)  # HSPF2


def test_1_speed_rating_version():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    dx_unit_2017 = RESNETDXModel(
        rated_net_cooling_cop=COP_C,
        rated_net_heating_cop=COP_H,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )
    dx_unit_2023 = RESNETDXModel(
        rated_net_cooling_cop=COP_C,
        rated_net_heating_cop=COP_H,
        rating_standard=AHRIVersion.AHRI_210_240_2023,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )

    assert dx_unit_2017.rated_cooling_airflow[0] == approx(
        dx_unit_2023.rated_cooling_airflow[0], 0.001
    )
    assert dx_unit_2017.seer() >= dx_unit_2023.seer()  # SEER > SEER2
    assert dx_unit_2017.hspf() > dx_unit_2023.hspf()  # HSPF > HSPF2


def test_2_speed_regression():
    # Two speed, SEER 17, HSPF 10
    seer_2 = 17.0
    hspf_2 = 10.0

    dx_unit_2_speed = make_rating_unit(
        StagingType.TWO_STAGE,
        seer_2,
        hspf_2,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )

    dx_unit_2_speed.print_cooling_info()

    dx_unit_2_speed.print_heating_info()
    dx_unit_2_speed.print_heating_info(region=2)
    assert dx_unit_2_speed.gross_total_cooling_capacity() == approx(
        dx_unit_2_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_2_speed.seer() == approx(seer_2, 0.01)
    assert dx_unit_2_speed.rated_net_cooling_cop[0] == approx(3.980, 0.001)
    assert dx_unit_2_speed.rated_net_cooling_cop[1] == approx(4.166, 0.001)
    assert dx_unit_2_speed.hspf() == approx(hspf_2, 0.01)
    assert dx_unit_2_speed.rated_net_heating_cop[0] == approx(3.347, 0.001)
    assert dx_unit_2_speed.rated_net_heating_cop[1] == approx(3.937, 0.001)


def test_neep_statistical_vchp_regression():
    # AHRI Certification #: 202680596 https://ashp.neep.org/#!/product/34439/7/25000/95/7500/0///0
    # SEER2 21, EER2 13, HSPF(4) = 11
    vchp_unit = RESNETDXModel(
        staging_type=StagingType.VARIABLE_SPEED,
        input_seer=21,
        input_eer=13,
        input_hspf=11,
        rated_net_total_cooling_capacity=fr_u(14000, "Btu/h"),
        rated_net_heating_capacity=fr_u(18000, "Btu/h"),
        rated_net_heating_capacity_17=fr_u(12100, "Btu/h"),
    )
    assert vchp_unit.net_total_cooling_capacity() == approx(
        vchp_unit.rated_net_total_cooling_capacity[vchp_unit.cooling_full_load_speed],
        0.01,
    )
    assert vchp_unit.net_total_cooling_cop() == approx(
        fr_u(vchp_unit.input_eer, "Btu/Wh"),
        0.01,
    )

    vchp_unit.print_cooling_info()

    assert vchp_unit.net_steady_state_heating_capacity() == approx(
        vchp_unit.rated_net_heating_capacity[vchp_unit.heating_full_load_speed],
        0.01,
    )

    vchp_unit.print_heating_info()

    assert vchp_unit.seer() == approx(vchp_unit.input_seer, 0.05)
    assert vchp_unit.eer() == approx(vchp_unit.input_eer, 0.01)
    assert vchp_unit.hspf() == approx(
        vchp_unit.input_hspf, 0.2
    )  # This match is bad because cooling capacity is so high relative to heating capacity


def test_neep_vchp_regression():
    # AHRI Certification #: 202680596 https://ashp.neep.org/#!/product/34439/7/25000/95/7500/0///0
    # SEER2 21, EER2 13, HSPF(4) = 11

    cooling_capacities = [
        [3428, None, 20098],  # 82
        [3100, 14000, 18200],  # 95
    ]

    cooling_powers = [
        [0.19, None, 1.8],  # 82
        [0.21, 1.08, 2.0],  # 95
    ]

    heating_capacities = [
        [2080, None, 14100],  # 5
        [2150, 12100, 16400],  # 17
        [4800, 18000, 20900],  # 47
    ]

    heating_powers = [
        [0.24, None, 1.57],  # 5
        [0.2, 1.6, 2.01],  # 17
        [0.2, 1.6, 2.01],  # 47
    ]

    vchp_unit = RESNETDXModel(
        tabular_data=make_neep_model_data(
            cooling_capacities, cooling_powers, heating_capacities, heating_powers
        ),
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )
    assert vchp_unit.net_total_cooling_capacity() == approx(
        vchp_unit.rated_net_total_cooling_capacity[vchp_unit.cooling_full_load_speed],
        0.01,
    )
    assert vchp_unit.net_total_cooling_cop() == approx(
        vchp_unit.rated_net_cooling_cop[vchp_unit.cooling_full_load_speed], 0.01
    )

    vchp_unit.print_cooling_info()

    assert vchp_unit.net_integrated_heating_capacity() == approx(
        vchp_unit.rated_net_heating_capacity[vchp_unit.heating_full_load_speed], 0.01
    )
    assert vchp_unit.net_integrated_heating_cop() == approx(
        vchp_unit.rated_net_heating_cop[vchp_unit.heating_full_load_speed], 0.01
    )
    vchp_unit.print_heating_info()
    vchp_unit.print_heating_info(region=2)

    assert vchp_unit.seer() == approx(16.901, 0.01)
    assert vchp_unit.eer() == approx(12.963, 0.01)
    assert vchp_unit.hspf() == approx(10.617, 0.01)
    assert vchp_unit.hspf(region=2) == approx(18.21, 0.01)


def test_plot():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    eer_1 = seer_1 / 1.2
    cop_1_h, _ = optimize.newton(
        lambda x: RESNETDXModel(
            rated_net_heating_cop=x,
            input_seer=seer_1,
            input_eer=eer_1,
            input_hspf=hspf_1,
            rating_standard=AHRIVersion.AHRI_210_240_2017,
        ).hspf()
        - hspf_1,
        hspf_1 / 2.0,
        full_output=True,
    )
    dx_unit_1_speed = RESNETDXModel(
        rated_net_heating_cop=cop_1_h,
        input_seer=seer_1,
        input_eer=eer_1,
        input_hspf=hspf_1,
        rating_standard=AHRIVersion.AHRI_210_240_2017,
    )

    dx_unit_1_speed.plot("output/heat-pump.html")

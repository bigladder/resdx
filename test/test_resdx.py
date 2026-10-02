from koozie import fr_u
from pytest import approx, mark, raises, warns
from scipy import optimize

from resdx import (
    RatingStandard,
    RESNETDXModel,
    RoomHeatPumpType,
    StagingType,
    make_neep_model_data,
    make_performance_map,
)
from resdx.defrost import DefrostStrategy
from resdx.models.tabular_data import make_two_speed_model_data, make_single_speed_model_data
from resdx.rating_solver import make_rating_unit

# Single speed gross COP values used for regression testing
COP_C = 3.312
COP_H = 3.051


# Tests
def test_1_speed_regression():
    # Single speed, SEER 13, HSPF 8
    seer = 13.0
    hspf = 8.0

    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        seer,
        hspf,
        rating_standard=RatingStandard.AHRI_210_240_2017,
    )

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.gross_total_cooling_capacity() == approx(
        dx_unit_1_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_1_speed.cooling_fan_power() == approx(dx_unit_1_speed.rated_cooling_fan_power[0], 0.0001)
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
        rating_standard=RatingStandard.AHRI_210_240_2017,
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
        rating_standard=RatingStandard.AHRI_210_240_2017,
    )

    dx_unit_1_speed.set_rating_standard(RatingStandard.AHRI_210_240_2023)

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
        rating_standard=RatingStandard.AHRI_210_240_2017,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )
    dx_unit_2023 = RESNETDXModel(
        rated_net_cooling_cop=COP_C,
        rated_net_heating_cop=COP_H,
        rating_standard=RatingStandard.AHRI_210_240_2023,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )

    assert dx_unit_2017.rated_cooling_airflow[0] == approx(dx_unit_2023.rated_cooling_airflow[0], 0.001)
    assert dx_unit_2017.seer() >= dx_unit_2023.seer()  # SEER > SEER2
    assert dx_unit_2017.hspf() > dx_unit_2023.hspf()  # HSPF > HSPF2


def test_1_speed_tabular_regression():
    seer_2 = 10.0
    eer_2 = 9.7
    hspf_2 = 7

    dx_unit_2_speed = RESNETDXModel(
        tabular_data=make_single_speed_model_data(
            cooling_capacity_95=fr_u(3.0, "ton_ref"),
            seer2=seer_2,
            eer2=eer_2,
            hspf2=hspf_2,
            heating_capacity_47=fr_u(3.0, "ton_ref"),
            heating_capacity_17=None,
        )
    )

    dx_unit_2_speed.print_cooling_info()

    dx_unit_2_speed.print_heating_info()
    dx_unit_2_speed.print_heating_info(region=2)
    assert dx_unit_2_speed.gross_total_cooling_capacity() == approx(
        dx_unit_2_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_2_speed.seer() == approx(seer_2, 0.01)
    assert dx_unit_2_speed.hspf() == approx(hspf_2, 0.01)


def test_2_speed_regression():
    # Two speed, SEER 17, HSPF 10
    seer_2 = 17.0
    hspf_2 = 10.0

    dx_unit_2_speed = make_rating_unit(
        StagingType.TWO_STAGE,
        seer_2,
        hspf_2,
        rating_standard=RatingStandard.AHRI_210_240_2017,
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
    assert dx_unit_2_speed.rated_net_heating_cop[0] == approx(3.395, 0.001)
    assert dx_unit_2_speed.rated_net_heating_cop[1] == approx(3.995, 0.001)


def test_2_speed_tabular_regression():
    # Two speed, SEER 17, HSPF 10
    seer_2 = 17.0
    hspf_2 = 10.0

    dx_unit_2_speed = RESNETDXModel(
        tabular_data=make_two_speed_model_data(
            cooling_capacity_95=fr_u(3.0, "ton_ref"),
            seer2=seer_2,
            eer2=seer_2 / 1.2,
            hspf2=hspf_2,
            heating_capacity_47=fr_u(3.0, "ton_ref"),
            heating_capacity_17=None,
        )
    )

    dx_unit_2_speed.print_cooling_info()

    dx_unit_2_speed.print_heating_info()
    dx_unit_2_speed.print_heating_info(region=2)
    assert dx_unit_2_speed.gross_total_cooling_capacity() == approx(
        dx_unit_2_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_2_speed.seer() == approx(seer_2, 0.01)
    assert dx_unit_2_speed.rated_net_cooling_cop[0] == approx(4.152, 0.001)
    assert dx_unit_2_speed.rated_net_cooling_cop[1] == approx(4.156, 0.001)
    assert dx_unit_2_speed.hspf() == approx(hspf_2, 0.01)
    assert dx_unit_2_speed.rated_net_heating_cop[0] == approx(4.577, 0.001)
    assert dx_unit_2_speed.rated_net_heating_cop[1] == approx(5.385, 0.001)


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


def test_1_speed_ceer():
    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        13.0,
        8.0,
        rating_standard=RatingStandard.AHRI_210_240_2017,
    )
    # Without standby power, single speed CEER is EER at 95 °F
    assert dx_unit_1_speed.ceer() == approx(dx_unit_1_speed.eer(dx_unit_1_speed.A_full_cond), 0.0001)

    # Standby power adds an equivalent power of 5115 h / 750 h times the inactive mode power
    dx_unit_1_speed.inactive_mode_power = 0.5
    q = dx_unit_1_speed.net_total_cooling_capacity(dx_unit_1_speed.A_full_cond)
    p = dx_unit_1_speed.net_cooling_power(dx_unit_1_speed.A_full_cond)
    assert fr_u(dx_unit_1_speed.ceer(), "Btu/Wh") == approx(q / (p + 0.5 * 5115.0 / 750.0), 0.0001)


def test_room_unit_rating_standard():
    room_unit = RESNETDXModel(
        rated_net_cooling_cop=COP_C,
        rated_net_heating_cop=COP_H,
        input_seer=13.0,
        input_hspf=8.0,
        is_ducted=False,
        rating_standard=RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024,
    )
    with raises(RuntimeError, match="Use ceer"):
        room_unit.seer()
    with raises(RuntimeError, match="Use heer"):
        room_unit.hspf()
    room_unit.print_cooling_info()  # Reports CEER
    room_unit.print_heating_info()  # Reports HEER

    room_unit.set_rating_standard(RatingStandard.AHRI_210_240_2023)
    assert room_unit.seer() > 0.0

    with raises(RuntimeError, match="non-ducted"):
        RESNETDXModel(
            rated_net_cooling_cop=COP_C,
            rated_net_heating_cop=COP_H,
            input_seer=13.0,
            input_hspf=8.0,
            is_ducted=True,
            rating_standard=RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024,
        )


def test_heer_regression():
    vchp_unit = make_rating_unit(StagingType.VARIABLE_SPEED, 21.0, 11.0, t_min=fr_u(-10.0, "°F"))
    assert vchp_unit.room_heat_pump_type() == RoomHeatPumpType.TYPE_4
    assert vchp_unit.heer() == approx(13.567, 0.001)

    # Room heat pump type follows cut-out/cut-in temperatures and active defrost
    vchp_unit.heating_off_temperature = vchp_unit.heating_on_temperature = fr_u(10.0, "°F")
    assert vchp_unit.room_heat_pump_type() == RoomHeatPumpType.TYPE_3
    assert vchp_unit.heer() == approx(11.875, 0.001)

    vchp_unit.heating_off_temperature = vchp_unit.heating_on_temperature = fr_u(30.0, "°F")
    assert vchp_unit.room_heat_pump_type() == RoomHeatPumpType.TYPE_2
    assert vchp_unit.heer() == approx(6.545, 0.001)

    vchp_unit.defrost.strategy = DefrostStrategy.NONE
    assert vchp_unit.room_heat_pump_type() == RoomHeatPumpType.TYPE_1
    assert vchp_unit.heer() == approx(3.796, 0.001)

    dx_unit_1_speed = make_rating_unit(StagingType.SINGLE_STAGE, 13.0, 8.0, t_min=fr_u(-10.0, "°F"))
    assert dx_unit_1_speed.heer() == approx(7.856, 0.001)

    dx_unit_2_speed = make_rating_unit(StagingType.TWO_STAGE, 17.0, 10.0, t_min=fr_u(-10.0, "°F"))
    assert dx_unit_2_speed.heer() == approx(11.087, 0.001)


@mark.filterwarnings("ignore:Low speed capacity at 82")
def test_room_unit_rating_solver():
    room_standard = RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024
    eer = 11.26  # Btu/Wh at 95 °F
    ceer = 16.8
    heer = 10.9

    # NEEP statistical low speed capacity is below the Appendix F range
    with warns(UserWarning, match="Appendix F requires 47% to 57%"):
        vchp_unit = make_rating_unit(
            StagingType.VARIABLE_SPEED,
            eer=eer,
            ceer=ceer,
            heer=heer,
            q95=fr_u(9000, "Btu/h"),
            t_min=fr_u(-20.0, "°F"),
            rating_standard=room_standard,
        )
    assert vchp_unit.ceer() == approx(ceer, 0.0001)
    assert vchp_unit.heer() == approx(heer, 0.0001)
    assert vchp_unit.eer(vchp_unit.A_full_cond) == approx(eer, 0.0001)

    # Without standby power, CEER reduces to a weighted average of EERs at the four test conditions
    eers = [
        vchp_unit.eer(condition)
        for condition in [
            vchp_unit.A_full_cond,
            vchp_unit.CEER_92_full_cond,
            vchp_unit.E_int_cond,
            vchp_unit.B_low_cond,
        ]
    ]
    weights = [0.08, 0.20, 0.33, 0.39]
    assert ceer == approx(sum(w * e for w, e in zip(weights, eers)) / 1.01194, 0.0001)

    dx_unit_2_speed = make_rating_unit(
        StagingType.TWO_STAGE, eer=9.0, ceer=12.0, heer=6.0, t_min=fr_u(10.0, "°F"), rating_standard=room_standard
    )
    assert dx_unit_2_speed.ceer() == approx(12.0, 0.0001)
    assert dx_unit_2_speed.heer() == approx(6.0, 0.0001)

    # Single speed EER is determined from CEER (including standby power)
    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE,
        ceer=11.0,
        heer=6.0,
        t_min=fr_u(10.0, "°F"),
        rating_standard=room_standard,
        inactive_mode_power=0.5,
    )
    assert dx_unit_1_speed.ceer() == approx(11.0, 0.0001)
    assert dx_unit_1_speed.heer() == approx(6.0, 0.0001)
    assert dx_unit_1_speed.eer(dx_unit_1_speed.A_full_cond) > 11.0

    with raises(ValueError, match="ceer and heer are required"):
        make_rating_unit(StagingType.VARIABLE_SPEED, eer=eer, ceer=ceer, rating_standard=room_standard)
    with raises(ValueError, match="seer and hspf are not used"):
        make_rating_unit(StagingType.VARIABLE_SPEED, 21.0, eer=eer, ceer=ceer, heer=heer, rating_standard=room_standard)
    with raises(ValueError, match="eer is required"):
        make_rating_unit(StagingType.VARIABLE_SPEED, ceer=ceer, heer=heer, rating_standard=room_standard)
    with raises(ValueError, match="ceer and heer are not used"):
        make_rating_unit(StagingType.VARIABLE_SPEED, 21.0, 11.0, ceer=ceer)


@mark.filterwarnings("ignore:Low speed capacity at 82")
def test_neep_statistical_vchp_high_cutout():
    # Heating off temperature above 5 °F (e.g., type 2 room heat pumps)
    vchp_unit = make_rating_unit(
        StagingType.VARIABLE_SPEED,
        eer=8.19,
        ceer=13.2,
        heer=5.8,
        q95=fr_u(9000, "Btu/h"),
        t_min=fr_u(30.0, "°F"),
        rating_standard=RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024,
    )
    assert vchp_unit.room_heat_pump_type() == RoomHeatPumpType.TYPE_2
    assert vchp_unit.ceer() == approx(13.2, 0.0001)
    assert vchp_unit.heer() == approx(5.8, 0.0001)


@mark.filterwarnings("ignore:Low speed capacity at 82")
def test_rating_unit_modify_map():
    t_95 = fr_u(95.0, "°F")

    def set_minimum_cooling_capacity_ratio(performance_map):
        performance_map.set_cooling_capacity_ratio(1, t_95, 0.5)

    # Ratings are matched with the map modifications applied
    vchp_unit = make_rating_unit(
        StagingType.VARIABLE_SPEED,
        eer=11.26,
        ceer=16.8,
        heer=10.9,
        q95=fr_u(9000, "Btu/h"),
        t_min=fr_u(-20.0, "°F"),
        rating_standard=RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024,
        modify_map=set_minimum_cooling_capacity_ratio,
    )
    capacities = vchp_unit.net_tabular_data.cooling_capacities
    assert capacities.get(1, t_95) / capacities.get(2, t_95) == approx(0.5, 0.0001)
    assert vchp_unit.ceer() == approx(16.8, 0.0001)
    assert vchp_unit.heer() == approx(10.9, 0.0001)


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
        tabular_data=make_neep_model_data(cooling_capacities, cooling_powers, heating_capacities, heating_powers),
        rating_standard=RatingStandard.AHRI_210_240_2017,
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
    assert vchp_unit.hspf(region=2) == approx(17.66, 0.01)


def test_make_performance_map():

    cooling_temperatures = [82.0, 95.0]
    cooling_capacities = [
        [1.0, 2.0, 3.0],  # 82
        [0.8, 1.6, 2.4],  # 95
    ]

    cooling_cops = [
        [4.0, 3.0, 2.0],  # 82
        [3.0, 2.0, 1.0],  # 95
    ]

    heating_temperatures = [17.0, 47.0]

    heating_capacities = [
        [0.8, 1.6, 2.4],  # 17
        [1.0, 2.0, 3.0],  # 47
    ]

    heating_cops = [
        [3.0, 2.0, 1.0],  # 17
        [4.0, 3.0, 2.0],  # 47
    ]

    tabular_data = make_performance_map(
        cooling_capacities, cooling_cops, cooling_temperatures, heating_capacities, heating_cops, heating_temperatures
    )

    speed_to_add = 2
    tabular_data.add_speed(speed_to_add)

    assert tabular_data.cooling_capacities.data[0][speed_to_add - 1] == approx(fr_u(1.5, "ton_ref"), 0.01)

    speed_to_add = 4
    tabular_data.add_speed(speed_to_add)
    assert tabular_data.cooling_capacities.data[0][speed_to_add - 1] == approx(fr_u(2.5, "ton_ref"), 0.01)


def test_plot():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    eer_1 = seer_1 / 1.2
    cop_1_h, _ = optimize.newton(
        lambda x: (
            RESNETDXModel(
                rated_net_heating_cop=x,
                input_seer=seer_1,
                input_eer=eer_1,
                input_hspf=hspf_1,
                rating_standard=RatingStandard.AHRI_210_240_2017,
            ).hspf()
            - hspf_1
        ),
        hspf_1 / 2.0,
        full_output=True,
    )
    dx_unit_1_speed = RESNETDXModel(
        rated_net_heating_cop=cop_1_h,
        input_seer=seer_1,
        input_eer=eer_1,
        input_hspf=hspf_1,
        rating_standard=RatingStandard.AHRI_210_240_2017,
    )

    dx_unit_1_speed.plot("output/heat-pump.html")


@mark.filterwarnings("ignore:Low speed capacity at 82")
def test_room_unit_plot():
    room_standard = RatingStandard.DOE_APPENDIX_F_ENERGY_STAR_2024
    vchp_unit = make_rating_unit(
        StagingType.VARIABLE_SPEED,
        eer=11.26,
        ceer=16.8,
        heer=10.9,
        q95=fr_u(9000, "Btu/h"),
        t_min=fr_u(-20.0, "°F"),
        rating_standard=room_standard,
    )
    vchp_unit.plot("output/room-heat-pump.html", show_seasonal_bins=True)

    # CEER test conditions are stored as rating bins
    assert len(vchp_unit.cooling_rating_bins.fractions) == 4
    assert sum(vchp_unit.cooling_rating_bins.fractions) == approx(1.0, 0.0001)

    dx_unit_1_speed = make_rating_unit(
        StagingType.SINGLE_STAGE, ceer=11.0, heer=6.0, t_min=fr_u(10.0, "°F"), rating_standard=room_standard
    )
    dx_unit_1_speed.ceer()
    assert len(dx_unit_1_speed.cooling_rating_bins.fractions) == 1

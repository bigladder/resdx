import resdx
from koozie import fr_u

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

from resdx.dx_unit import AHRIVersion

sns.set()

import numpy as np

from pytest import approx


DXUnit = resdx.DXUnit
StagingType = resdx.StagingType
PsychState = resdx.psychrometrics.PsychState
HeatingConditions = resdx.dx_unit.HeatingConditions
CoolingConditions = resdx.dx_unit.CoolingConditions


# Single speed gross COP values used for regression testing
COP_C = 4.277
COP_H = 3.752


# Tests
def test_1_speed_regression():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    cop_1_c, solution_1_c = optimize.newton(
        lambda x: DXUnit(rated_gross_cooling_cop=x, input_seer=seer_1).seer() - seer_1,
        seer_1 / 3.33,
        full_output=True,
    )
    cop_1_h, solution_1_h = optimize.newton(
        lambda x: DXUnit(rated_gross_heating_cop=x, input_hspf=hspf_1).hspf() - hspf_1,
        hspf_1 / 2.0,
        full_output=True,
    )
    dx_unit_1_speed = DXUnit(
        rated_gross_cooling_cop=cop_1_c,
        rated_gross_heating_cop=cop_1_h,
        input_seer=seer_1,
        input_hspf=hspf_1,
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
    assert dx_unit_1_speed.seer() == approx(seer_1, 0.0001)
    assert dx_unit_1_speed.hspf() == approx(hspf_1, 0.0001)
    assert dx_unit_1_speed.rated_gross_cooling_cop[0] == approx(COP_C, 0.001)
    assert dx_unit_1_speed.rated_gross_heating_cop[0] == approx(COP_H, 0.001)


def test_1_speed_refrigerant_charge_regression():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    cop_1_c, solution_1_c = optimize.newton(
        lambda x: DXUnit(rated_gross_cooling_cop=x, input_seer=seer_1).seer() - seer_1,
        seer_1 / 3.33,
        full_output=True,
    )
    cop_1_h, solution_1_h = optimize.newton(
        lambda x: DXUnit(rated_gross_heating_cop=x, input_hspf=hspf_1).hspf() - hspf_1,
        hspf_1 / 2.0,
        full_output=True,
    )
    dx_unit_1_speed = DXUnit(
        rated_gross_cooling_cop=cop_1_c,
        rated_gross_heating_cop=cop_1_h,
        input_seer=seer_1,
        input_hspf=hspf_1,
        refrigerant_charge_deviation=-0.25,
    )

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.seer() == approx(11.16, 0.01)
    assert dx_unit_1_speed.hspf() == approx(7.05, 0.01)


def test_1_speed_2023_regression():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    dx_unit_1_speed = DXUnit(
        rated_gross_cooling_cop=COP_C,
        rated_gross_heating_cop=COP_H,
        rating_standard=AHRIVersion.AHRI_210_240_2023,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )

    dx_unit_1_speed.print_cooling_info()
    dx_unit_1_speed.print_heating_info()
    assert dx_unit_1_speed.seer() == approx(12.999, 0.001)  # SEER2
    assert dx_unit_1_speed.hspf() == approx(7.059, 0.001)  # HSPF2


def test_1_speed_rating_version():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    dx_unit_2017 = DXUnit(
        rated_gross_cooling_cop=COP_C,
        rated_gross_heating_cop=COP_H,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )
    dx_unit_2023 = DXUnit(
        rated_gross_cooling_cop=COP_C,
        rated_gross_heating_cop=COP_H,
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
    cop_2_c, solution_2_c = optimize.newton(
        lambda x: DXUnit(
            staging_type=resdx.StagingType.TWO_STAGE,
            rated_gross_cooling_cop=x,
            input_seer=seer_2,
        ).seer()
        - seer_2,
        seer_2 / 3.33,
        full_output=True,
    )
    cop_2_h, solution_2_h = optimize.newton(
        lambda x: DXUnit(
            staging_type=resdx.StagingType.TWO_STAGE,
            rated_gross_heating_cop=x,
            input_hspf=hspf_2,
        ).hspf()
        - hspf_2,
        hspf_2 / 2.0,
        full_output=True,
    )

    dx_unit_2_speed = DXUnit(
        staging_type=resdx.StagingType.TWO_STAGE,
        rated_gross_cooling_cop=cop_2_c,
        rated_gross_heating_cop=cop_2_h,
        input_seer=seer_2,
        input_hspf=hspf_2,
    )

    dx_unit_2_speed.print_cooling_info()

    dx_unit_2_speed.print_heating_info()
    dx_unit_2_speed.print_heating_info(region=2)
    assert dx_unit_2_speed.gross_total_cooling_capacity() == approx(
        dx_unit_2_speed.rated_gross_total_cooling_capacity[0], 0.01
    )
    assert dx_unit_2_speed.seer() == approx(seer_2, 0.01)
    assert dx_unit_2_speed.rated_gross_cooling_cop[0] == approx(4.343, 0.001)
    assert dx_unit_2_speed.rated_gross_cooling_cop[1] == approx(4.772, 0.001)
    assert dx_unit_2_speed.hspf() == approx(hspf_2, 0.01)
    assert dx_unit_2_speed.rated_gross_heating_cop[0] == approx(4.020, 0.001)
    assert dx_unit_2_speed.rated_gross_heating_cop[1] == approx(4.621, 0.001)


def test_vchp_regression():
    # VCHP (Fujitsu Halcyon 12) https://ashp.neep.org/#!/product/25349/7/25000///0
    # SEER 21.3, EER 13.4, HSPF(4) = 11.7
    cooling_data = resdx.VCHPDataPoints()
    cooling_data.append(
        resdx.VCHPDataPoint(
            drybulb=fr_u(95.0, "°F"),
            capacities=[fr_u(13600, "Btu/h"), fr_u(3100, "Btu/h")],
            cops=[2.75, 7.57],
        )
    )
    cooling_data.append(
        resdx.VCHPDataPoint(
            drybulb=fr_u(82.0, "°F"),
            capacities=[fr_u(15276, "Btu/h"), fr_u(3437, "Btu/h")],
            cops=[3.2, 8.39],
        )
    )

    heating_data = resdx.VCHPDataPoints()
    heating_data.append(
        resdx.VCHPDataPoint(
            drybulb=fr_u(47.0, "°F"),
            capacities=[fr_u(19400, "Btu/h"), fr_u(3100, "Btu/h")],
            cops=[3.09, 6.49],
        )
    )
    heating_data.append(
        resdx.VCHPDataPoint(
            drybulb=fr_u(17.0, "°F"),
            capacities=[fr_u(17600, "Btu/h"), fr_u(2824, "Btu/h")],
            cops=[2.62, 5.52],
        )
    )
    heating_data.append(
        resdx.VCHPDataPoint(
            drybulb=fr_u(5.0, "°F"),
            capacities=[fr_u(16710, "Btu/h"), fr_u(2671, "Btu/h")],
            cops=[2.37, 4.89],
        )
    )

    vchp_unit = resdx.make_vchp_unit(cooling_data, heating_data)
    assert vchp_unit.net_total_cooling_capacity() == approx(
        vchp_unit.rated_net_total_cooling_capacity[0], 0.01
    )
    assert vchp_unit.net_cooling_cop() == approx(
        vchp_unit.rated_net_cooling_cop[0], 0.01
    )

    vchp_unit.print_cooling_info()

    assert vchp_unit.net_integrated_heating_capacity() == approx(
        vchp_unit.rated_net_heating_capacity[0], 0.01
    )
    assert vchp_unit.net_integrated_heating_cop() == approx(
        vchp_unit.rated_net_heating_cop[0], 0.01
    )
    vchp_unit.print_heating_info()
    vchp_unit.print_heating_info(region=2)

    assert vchp_unit.seer() == approx(20.292, 0.01)
    assert vchp_unit.eer() == approx(9.38338, 0.01)
    assert vchp_unit.hspf() == approx(13.21, 0.01)
    assert vchp_unit.hspf(region=2) == approx(13.68, 0.01)


def test_neep_vchp_regression():
    # SEER 21.3, EER 13.4, HSPF(4) = 11.7
    vchp_unit = DXUnit(
        staging_type=StagingType.VARIABLE_SPEED,
        input_seer=21.3,
        input_eer=13.4,
        input_hspf=11.7,
        rating_standard=AHRIVersion.AHRI_210_240_2023,
    )
    assert vchp_unit.net_total_cooling_capacity() == approx(
        vchp_unit.rated_net_total_cooling_capacity[vchp_unit.cooling_full_load_speed],
        0.01,
    )
    assert vchp_unit.net_cooling_cop() == approx(
        fr_u(vchp_unit.input_eer, "Btu/Wh"),
        0.01,
    )

    vchp_unit.print_cooling_info()

    assert vchp_unit.net_steady_state_heating_capacity() == approx(
        vchp_unit.rated_net_heating_capacity[vchp_unit.heating_full_load_speed],
        0.01,
    )

    vchp_unit.print_heating_info()

    assert vchp_unit.seer() == approx(19.44, 0.01)
    assert vchp_unit.eer() == approx(vchp_unit.input_eer, 0.01)
    assert vchp_unit.hspf() == approx(10.37, 0.01)


def test_plot():
    # Single speed, SEER 13, HSPF 8
    seer_1 = 13.0
    hspf_1 = 8.0
    cop_1_c, solution_1_c = optimize.newton(
        lambda x: DXUnit(rated_gross_cooling_cop=x, input_seer=seer_1).seer() - seer_1,
        seer_1 / 3.33,
        full_output=True,
    )
    cop_1_h, solution_1_h = optimize.newton(
        lambda x: DXUnit(rated_gross_heating_cop=x, input_hspf=hspf_1).hspf() - hspf_1,
        hspf_1 / 2.0,
        full_output=True,
    )
    dx_unit_1_speed = DXUnit(
        rated_gross_cooling_cop=cop_1_c,
        rated_gross_heating_cop=cop_1_h,
        input_seer=seer_1,
        input_hspf=hspf_1,
    )

    # Plot integrated power and capacity
    T_out = np.arange(-23, 76, 1)
    conditions = [
        dx_unit_1_speed.make_condition(
            HeatingConditions, outdoor=PsychState(drybulb=fr_u(T, "°F"), rel_hum=0.4)
        )
        for T in T_out
    ]
    Q_integrated = [
        dx_unit_1_speed.gross_integrated_heating_capacity(condition)
        for condition in conditions
    ]
    P_integrated = [
        dx_unit_1_speed.gross_integrated_heating_power(condition)
        for condition in conditions
    ]
    COP_integrated = [
        dx_unit_1_speed.gross_integrated_heating_cop(condition)
        for condition in conditions
    ]

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Temp (°F)")
    ax1.set_ylabel("Capacity/Power (W)", color=color)
    ax1.plot(T_out, Q_integrated, color=color)
    ax1.plot(T_out, P_integrated, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim([0, 15000])

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("COP", color=color)
    ax2.plot(T_out, COP_integrated, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim([0, 5.5])

    fig.tight_layout()
    plt.savefig("output/heat-pump.png")

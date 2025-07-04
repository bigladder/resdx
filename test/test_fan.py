from koozie import fr_u
from pytest import approx

import resdx


def test_psc_fan():
    design_external_static_pressure = fr_u(0.5, "in_H2O")

    psc_fan = resdx.PSCFan(
        design_airflow=[fr_u(v, "cfm") for v in [1179.0, 1003.0, 740.0]],
        design_external_static_pressure=design_external_static_pressure,
        design_specific_fan_power=fr_u(0.33, "W/cfm"),
    )

    # Open flow conditions
    assert psc_fan.rotational_speed(0, fr_u(0.0, "in_H2O")) == fr_u(1040.0, "rpm")

    # Design conditions
    assert psc_fan.airflow(0, design_external_static_pressure) == fr_u(1179.0, "cfm")

    # Simple regression check
    assert psc_fan.power(2, fr_u(0.3, "in_H2O")) == approx(fr_u(248.04, "W"), 0.01)

    # Blocked fan
    assert psc_fan.airflow(0, fr_u(1.0, "in_H2O")) == fr_u(0.0, "cfm")

    # System curve checks
    assert psc_fan.airflow(0) == fr_u(1179.0, "cfm")
    assert psc_fan.airflow(1) == approx(fr_u(1061.045, "cfm"), 0.01)

    psc_fan.add_speed(psc_fan.design_airflow[0], fr_u(0.15, "in_H2O"))
    assert psc_fan.airflow(psc_fan.number_of_speeds - 1, fr_u(0.15, "in_H2O")) == psc_fan.design_airflow[0]


def test_ecm_fan():
    design_external_static_pressure = fr_u(0.5, "in_H2O")

    """Based on Fan #1 in Proctor measurements"""
    ecm_fan = resdx.ECMFlowFan(
        design_airflow=[fr_u(v, "cfm") for v in [2405.0, 2200.0, 1987.0, 1760.0, 1537.0, 1310.0, 1169.0, 1099.0]],
        design_external_static_pressure=design_external_static_pressure,
        design_specific_fan_power=fr_u(0.3665, "W/cfm"),
        maximum_power=fr_u(1000, "W"),
    )

    # Open flow conditions
    assert ecm_fan.airflow(0, fr_u(0.0, "in_H2O")) == fr_u(2405.0, "cfm")

    # Design conditions
    assert ecm_fan.rotational_speed(0, design_external_static_pressure) == fr_u(1100.0, "rpm")

    # Power limit
    assert ecm_fan.power(0, fr_u(0.8, "in_H2O")) == fr_u(1000.0, "W")

    # Simple regression check
    assert ecm_fan.power(2, fr_u(0.3, "in_H2O")) == approx(fr_u(474.119, "W"), 0.01)

    # System curve checks
    assert ecm_fan.airflow(0) == fr_u(2405.0, "cfm")
    assert ecm_fan.airflow(1) == fr_u(2200.0, "cfm")
    assert ecm_fan.power(0) == approx(fr_u(881.43, "W"), 0.01)
    assert ecm_fan.power(6) == approx(fr_u(102.08, "W"), 0.01)


def test_eere_fans():
    design_external_static_pressure = fr_u(0.0, "in_H2O")

    for fan_type in [
        resdx.EEREBaselinePSCFan,
        resdx.EEREImprovedPSCFan,
        resdx.EEREPSCWithControlsFan,
        resdx.EEREBPMSingleStageConstantTorqueFan,
        resdx.EEREBPMMultiStageConstantTorqueFan,
        resdx.EEREBPMMultiStageConstantAirflowFan,
        resdx.EEREBPMMultiStageBackwardCurvedImpellerConstantAirflowFan,
    ]:
        fan = fan_type(
            design_airflow=[fr_u(v, "cfm") for v in [1179.0, 1003.0, 740.0]],
            design_external_static_pressure=design_external_static_pressure,
        )
        assert fan.airflow(0, 0.0) == fr_u(1179.0, "cfm")


def test_resnet_fans():
    airflows = [fr_u(v * 100 + 1, "cfm") for v in reversed(range(11))]

    psc_fan = resdx.RESNETPSCFan(airflows)
    assert psc_fan.specific_fan_power(0) == psc_fan.design_specific_fan_power
    assert psc_fan.specific_fan_power(10) / psc_fan.specific_fan_power(0) == approx(
        1.0 - psc_fan.SPECIFIC_FAN_POWER_SLOPE, 0.01
    )

    bpm_fan = resdx.RESNETBPMFan(airflows)

    # Ducted fan
    assert bpm_fan.specific_fan_power(0, fr_u(0.5, "in_H2O")) == approx(bpm_fan.DUCTED_DESIGN_SPECIFIC_FAN_POWER, 1e-5)

    # Ductless fan
    assert bpm_fan.specific_fan_power(0, 0.0) == approx(bpm_fan.DUCTLESS_DESIGN_SPECIFIC_FAN_POWER, 1e-5)

    # Ductless BPM specific fan power ratio <= Ducted BPM specific fan power <= PSC specific fan power
    for speed in range(11):
        ductless_bpm_specific_fan_power_ratio = bpm_fan.specific_fan_power(speed, 0.0) / bpm_fan.specific_fan_power(
            0, 0.0
        )
        ducted_bpm_specific_fan_power_ratio = bpm_fan.specific_fan_power(speed) / bpm_fan.specific_fan_power(0)
        psc_specific_fan_power_ratio = psc_fan.specific_fan_power(speed) / psc_fan.specific_fan_power(0)
        assert ductless_bpm_specific_fan_power_ratio <= ducted_bpm_specific_fan_power_ratio
        assert ducted_bpm_specific_fan_power_ratio <= psc_specific_fan_power_ratio

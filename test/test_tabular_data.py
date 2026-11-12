from pytest import approx

from resdx.models.tabular_data import TemperatureSpeedPerformanceTable


def test_add_temperature_interpolation():
    # Values are linear in temperature, so an added temperature should land on the line
    table = TemperatureSpeedPerformanceTable(
        temperatures=[250.0, 260.0, 270.0, 280.0],
        number_of_speeds=2,
        data_in=[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]],
    )
    table.add_temperature(275.0)
    assert table.temperatures == [250.0, 260.0, 270.0, 275.0, 280.0]
    assert table.get(1, 275.0) == approx(3.5)
    assert table.get(2, 275.0) == approx(7.0)

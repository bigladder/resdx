from pytest import approx

from koozie import fr_u

from resdx import make_neep_statistical_model_data


def test_make_neep_statistical_model_data():

    size = fr_u(5.0, "ton_ref")
    eer2 = 11.0
    neep_model = make_neep_statistical_model_data(
        cooling_capacity_95=size,
        seer2=19.0,
        eer2=eer2,
        heating_capacity_47=size,
        heating_capacity_17=size * 0.7,
        hspf2=8.0,
    )
    assert neep_model.cooling_capacity() == approx(size, 0.0001)
    assert neep_model.cooling_cop() == approx(fr_u(eer2, "Btu/Wh"), 0.0001)
    assert neep_model.heating_capacity() == approx(size, 0.0001)

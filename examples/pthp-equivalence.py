# Used to convert PTHP ratings to SEER/HSPF

# %%
from resdx import RESNETDXModel, AHRIVersion, fr_u
from resdx.models.tabular_data import make_packaged_terminal_model_data

output_directory_path = "output"

cooling_capacity = 9700
cooling_eer = 12.2
heating_capacity = 8100
heating_cop = 3.7

dx_unit = RESNETDXModel(
    is_ducted=False,
    rated_net_cooling_cop=fr_u(cooling_eer, "Btu/(W*h)"),
    rated_net_heating_cop=heating_cop,
    rated_net_total_cooling_capacity=[fr_u(cooling_capacity, "Btu/h")],
    rated_net_heating_capacity=[fr_u(heating_capacity, "Btu/h")],
    rating_standard=AHRIVersion.AHRI_210_240_2017,
)

dx_unit.print_cooling_info()
dx_unit.print_heating_info()
dx_unit.plot(f"{output_directory_path}/pthp-traditional.html")

# %%
dx_unit = RESNETDXModel(
    is_ducted=False,
    tabular_data=make_packaged_terminal_model_data(
        cooling_capacity_95=fr_u(cooling_capacity, "Btu/h"),
        eer_95=cooling_eer,
        heating_capacity_47=fr_u(heating_capacity, "Btu/h"),
        heating_cop_47=heating_cop,
    ),
    rating_standard=AHRIVersion.AHRI_210_240_2017,
)

dx_unit.print_cooling_info()
dx_unit.print_heating_info()
dx_unit.plot(f"{output_directory_path}/pthp-tabular.html")

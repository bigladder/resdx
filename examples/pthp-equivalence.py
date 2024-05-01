# Used to convert PTHP ratings to SEER/HSPF

# %%
import resdx

AHRIVersion = resdx.AHRIVersion

fr_u = resdx.fr_u

cooling_capacity = 9600
cooling_eer = 12.2
heating_capacity = 8000
heating_cop = 3.7

dx_unit = resdx.DXUnit(
    is_ducted=False,
    rated_net_cooling_cop=fr_u(cooling_eer, "Btu/(W*h)"),
    rated_net_heating_cop=heating_cop,
    rated_net_total_cooling_capacity=fr_u(cooling_capacity, "Btu/h"),
    rated_net_heating_capacity=fr_u(heating_capacity, "Btu/h"),
    rating_standard=AHRIVersion.AHRI_210_240_2017,
)

dx_unit.print_cooling_info()
dx_unit.print_heating_info()

# %%

from .nrel import cutler_gross_cooling_power, cutler_gross_total_cooling_capacity, \
                  energyplus_gross_sensible_cooling_capacity, cutler_gross_steady_state_heating_capacity, \
                  epri_gross_integrated_heating_capacity, cutler_gross_steady_state_heating_power, \
                  epri_gross_integrated_heating_power

from .title24 import title24_shr

resnet_gross_cooling_power = cutler_gross_cooling_power
resnet_gross_total_cooling_capacity = cutler_gross_total_cooling_capacity
resnet_gross_sensible_cooling_capacity = energyplus_gross_sensible_cooling_capacity
resnet_shr_rated = title24_shr
resnet_gross_steady_state_heating_capacity = cutler_gross_steady_state_heating_capacity
resnet_gross_integrated_heating_capacity = epri_gross_integrated_heating_capacity
resnet_gross_steady_state_heating_power = cutler_gross_steady_state_heating_power
resnet_gross_integrated_heating_power = epri_gross_integrated_heating_power
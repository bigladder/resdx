'''
This file is generated from a template. To modify, edit the *.py.jinja file.
'''
from scipy.interpolate import RegularGridInterpolator

from ..enums import StagingType


def cop_47_h1_full(
    staging_type: StagingType, hspf: float, capacity_maintenance_17: float
) -> float:
    if staging_type == StagingType.SINGLE_STAGE:
        rating_range = [5.0, 6.5, 8.0, 9.5, 11.0]
        ratio_range = [0.5, 0.5333333333333333, 0.6, 0.7333333333333333, 1.0]
        input_data = [[1.9707525689167475, 2.8444383206836372, 3.9326226591917015, 5.326991657059977, 7.177930054093417], [1.9628443254473282, 2.800551694515633, 3.8193107805482707, 5.084928332811592, 6.699499297907473], [1.945849139352892, 2.720493249326752, 3.6215925755201135, 4.682863488723971, 5.9511778179594526], [1.9150348596070907, 2.5886355631293076, 3.3180810804373118, 4.110616451150018, 4.974796610575278], [1.9042669937369379, 2.4979555247233622, 3.1024902665758662, 3.71817118666892, 4.345309416663822]]
    elif staging_type == StagingType.TWO_STAGE:
        rating_range = [5.0, 6.5, 8.0, 9.5, 11.0]
        ratio_range = [0.5, 0.5333333333333333, 0.6, 0.7333333333333333, 1.0]
        input_data = [[1.7908856832596507, 2.586781714995416, 3.5758024515454507, 4.84262666405325, 6.523426061510016], [1.77541845104312, 2.5352521639685217, 3.4569741750052896, 4.601648668384456, 6.061299221207062], [1.752724985812227, 2.4510469309789586, 3.2624773122950987, 4.217871894025266, 5.3592735577279464], [1.7160437881453856, 2.3194022408453043, 2.972636318363227, 3.682196978127989, 4.455697799121371], [1.6539168354304852, 2.1693762782338726, 2.69416581967213, 3.228541093151714, 3.7727671568829297]]
    elif staging_type == StagingType.VARIABLE_SPEED:
        # Historic NEEP correlation: heating_cop_47 = 2.837 + 0.066 * hspf2
        rating_range = [7.0, 9.25, 11.5, 13.75, 16.0]
        ratio_range = [0.5, 0.54, 0.6200000000000001, 0.7800000000000001, 1.1]
        input_data = [[2.70162733300162, 4.049585217357317, 5.795797333021919, 8.202423537269693, 11.688619622795512], [2.619863515254638, 3.8322414514425156, 5.341042508946027, 7.264665909305705, 9.80042855993867], [2.4927066682076604, 3.5074512367329875, 4.6627116304483, 5.989686540550063, 7.529483300523657], [2.3643583391073655, 3.1680836578940763, 3.994575344387272, 4.8447715247557595, 5.719659094859766], [2.1822381843720597, 2.8749906050941996, 3.5635394941419962, 4.247896748325269, 4.928074443938942]]

    return float(RegularGridInterpolator((ratio_range, rating_range), input_data, "linear")(
        (capacity_maintenance_17, hspf)
    ))


def cop_82_b_low(
    staging_type: StagingType, seer: float, seer_eer_ratio: float
) -> float:
    if staging_type == StagingType.SINGLE_STAGE:
        raise RuntimeError("COP 82 B low is not available for single speed equipment.")
    elif staging_type == StagingType.TWO_STAGE:
        rating_range = [6.0, 22.0]
        ratio_range = [1.0, 2.4]
        input_data = [[1.7772312730848057, 6.51651466797766], [2.1046299727964275, 7.716976566920232]]
    elif staging_type == StagingType.VARIABLE_SPEED:
        # Historic NEEP correlation: EIRr82min = bracket(1.305 - 0.324 * seer2 / eer2, 0.2, 1.0)
        rating_range = [14.0, 24.5, 35.0]
        ratio_range = [1.0, 1.7466666666666666, 2.12, 2.3066666666666666, 2.4]
        input_data = [[4.0472407657688985, 7.061466285336162, 10.057968980927825], [6.174580645705325, 10.288519440812733, 14.052861468542359], [14.24024173044029, 23.261833752330144, 30.96218837002951], [19.50808827608118, 31.8424618079649, 42.38831468300697], [23.028778414477685, 37.51287887399246, 49.86316799389444]]

    return float(RegularGridInterpolator((ratio_range, rating_range), input_data, "linear")(
        (seer_eer_ratio, seer)
    ))
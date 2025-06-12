from dataclasses import dataclass


@dataclass
class RegressionResult:
    slope: float
    intercept: float


class StatisticalSet:
    def __init__(self, statistical_data: dict[str, dict[str]]):
        quantities = statistical_data.get("quantities", {})
        regressions = statistical_data.get("regressions", {})

        for quantity_name, quantity_statistics in quantities.items():
            setattr(self, quantity_name, quantity_statistics["mean"])

        for regression_name, regression_statistics in regressions.items():
            regression_result = RegressionResult(
                slope=regression_statistics["slope"], intercept=regression_statistics["intercept"]
            )
            setattr(self, regression_name, regression_result)


original_statistics = StatisticalSet(
    {
        "quantities": {
            "Qr95Rated": {"mean": 0.934},
            "Qm95Max": {"mean": 0.940},
            "Qm95Min": {"mean": 0.948},
            "EIRr95Rated": {"mean": 0.928},
            "EIRm95Max": {"mean": 1.326},
            "EIRm95Min": {"mean": 1.315},
            "Qr47Rated": {"mean": 0.908},
            "Qr47Min": {"mean": 0.272},
            "Qr17Rated": {"mean": 0.817},
            "Qr17Min": {"mean": 0.341},
            "Qm17Rated": {"mean": 0.689},
            "Qm5Max": {"mean": 0.866},
            "Qr5Rated": {"mean": 0.988},
            "Qr5Min": {"mean": 0.321},
            "QmslopeLCTMax": {"mean": -0.025},
            "QmslopeLCTMin": {"mean": -0.024},
            "EIRr47Rated": {"mean": 0.939},
            "EIRr47Min": {"mean": 0.730},
            "EIRm17Rated": {"mean": 1.351},
            "EIRr17Rated": {"mean": 0.902},
            "EIRr17Min": {"mean": 0.798},
            "EIRm5Max": {"mean": 1.164},
            "EIRr5Rated": {"mean": 1.000},
            "EIRr5Min": {"mean": 0.866},
            "EIRmslopeLCTMax": {"mean": 0.012},
            "EIRmslopeLCTMin": {"mean": 0.012},
        },
        "regressions": {
            "EIRr82MinvSEER2_over_EER2": {"slope": -0.324, "intercept": 1.305},
            "Qr95MinvSEER2_over_EER2": {"slope": -0.119, "intercept": 0.510},
        },
    }
)

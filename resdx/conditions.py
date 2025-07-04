from koozie import fr_u

from .psychrometrics import STANDARD_CONDITIONS, PsychState


class OperatingConditions:
    rated_net_capacity: float
    rated_standard_volumetric_airflow: float
    rated_volumetric_airflow: float
    rated_mass_airflow: float

    standard_volumetric_airflow: float
    volumetric_airflow: float
    mass_airflow: float
    mass_airflow_ratio: float
    standard_volumetric_airflow_per_capacity: float

    external_static_pressure: float

    def __init__(
        self,
        outdoor=STANDARD_CONDITIONS,
        indoor=STANDARD_CONDITIONS,
        compressor_speed=0,
        fan_speed=None,
        rated_flow_external_static_pressure=None,
    ):
        self.outdoor = outdoor
        self.indoor = indoor
        self.compressor_speed = compressor_speed  # compressor speed index (0 = full speed, 1 = next lowest, ...)
        self.rated_airflow_set = False
        self.rated_flow_external_static_pressure = rated_flow_external_static_pressure
        self.fan_speed = fan_speed

    def set_rated_volumetric_airflow(self, rated_volumetric_airflow, rated_net_capacity):
        self.rated_net_capacity = rated_net_capacity
        self.rated_standard_volumetric_airflow = rated_volumetric_airflow
        self.rated_volumetric_airflow = (
            self.rated_standard_volumetric_airflow * STANDARD_CONDITIONS.rho / self.indoor.rho
        )
        self.rated_mass_airflow = self.rated_volumetric_airflow * self.indoor.rho
        self.rated_airflow_set = True
        self.set_volumetric_airflow(self.rated_volumetric_airflow)

    def set_volumetric_airflow(self, volumetric_airflow):
        self.volumetric_airflow = volumetric_airflow
        self.mass_airflow = volumetric_airflow * self.indoor.rho
        self.standard_volumetric_airflow = self.mass_airflow / STANDARD_CONDITIONS.rho
        if self.rated_airflow_set:
            self.mass_airflow_ratio = self.mass_airflow / self.rated_mass_airflow
            self.standard_volumetric_airflow_per_capacity = self.standard_volumetric_airflow / self.rated_net_capacity
            if self.rated_flow_external_static_pressure is not None:
                self.external_static_pressure = (
                    self.rated_flow_external_static_pressure * (self.mass_airflow_ratio) ** 2.0
                )

    def set_standard_volumetric_airflow(self, volumetric_airflow):
        self.standard_volumetric_airflow = volumetric_airflow
        self.volumetric_airflow = self.standard_volumetric_airflow * STANDARD_CONDITIONS.rho / self.indoor.rho
        self.mass_airflow = self.volumetric_airflow * self.indoor.rho
        if self.rated_airflow_set:
            self.mass_airflow_ratio = self.mass_airflow / self.rated_mass_airflow
            self.standard_volumetric_airflow_per_capacity = self.standard_volumetric_airflow / self.rated_net_capacity
            if self.rated_flow_external_static_pressure is not None:
                self.external_static_pressure = (
                    self.rated_flow_external_static_pressure * (self.mass_airflow_ratio) ** 2.0
                )

    def set_mass_airflow(self, mass_airflow):
        self.mass_airflow = mass_airflow
        if self.rated_flow_external_static_pressure is not None:
            self.external_static_pressure = self.rated_flow_external_static_pressure * (self.mass_airflow_ratio) ** 2.0
        if self.rated_airflow_set:
            self.mass_airflow_ratio = self.rated_mass_airflow / self.mass_airflow
            self.volumetric_airflow = self.mass_airflow / self.indoor.rho
            self.standard_volumetric_airflow = self.mass_airflow / STANDARD_CONDITIONS.rho
            self.standard_volumetric_airflow_per_capacity = self.standard_volumetric_airflow / self.rated_net_capacity

    def set_mass_airflow_ratio(self, mass_airflow_ratio):
        self.mass_airflow_ratio = mass_airflow_ratio
        if self.rated_flow_external_static_pressure is not None:
            self.external_static_pressure = self.rated_flow_external_static_pressure * (self.mass_airflow_ratio) ** 2.0
        if self.rated_airflow_set:
            self.mass_airflow = self.mass_airflow_ratio * self.rated_mass_airflow
            self.volumetric_airflow = self.mass_airflow / self.indoor.rho
            self.standard_volumetric_airflow = self.mass_airflow / STANDARD_CONDITIONS.rho
            self.standard_volumetric_airflow_per_capacity = self.standard_volumetric_airflow / self.rated_net_capacity

    def set_new_fan_speed(self, fan_speed, airflow):
        self.fan_speed = fan_speed
        self.set_standard_volumetric_airflow(airflow)


class CoolingConditions(OperatingConditions):
    def __init__(
        self,
        outdoor=PsychState(drybulb=fr_u(95.0, "°F"), wetbulb=fr_u(75.0, "°F")),
        indoor=PsychState(drybulb=fr_u(80.0, "°F"), wetbulb=fr_u(67.0, "°F")),
        compressor_speed=0,
        fan_speed=None,
        rated_flow_external_static_pressure=fr_u(0.2, "in_H2O"),
    ):
        super().__init__(
            outdoor,
            indoor,
            compressor_speed,
            fan_speed,
            rated_flow_external_static_pressure,
        )


class HeatingConditions(OperatingConditions):
    def __init__(
        self,
        outdoor=PsychState(drybulb=fr_u(47.0, "°F"), wetbulb=fr_u(43.0, "°F")),
        indoor=PsychState(drybulb=fr_u(70.0, "°F"), wetbulb=fr_u(60.0, "°F")),
        compressor_speed=0,
        fan_speed=None,
        rated_flow_external_static_pressure=fr_u(0.2, "in_H2O"),
    ):
        super().__init__(
            outdoor,
            indoor,
            compressor_speed,
            fan_speed,
            rated_flow_external_static_pressure,
        )

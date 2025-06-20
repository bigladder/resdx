import psychrolib
from koozie import fr_u, to_u

psychrolib.SetUnitSystem(psychrolib.SI)


class PsychState:
    def __init__(self, drybulb, pressure=fr_u(1.0, "atm"), **kwargs):
        self.db = drybulb
        self.db_C = to_u(self.db, "°C")
        self.p = pressure
        self._wb = -999.0
        self._rho = -999.0
        self._h = -999.0
        self._rh = -999.0
        self._hr = -999.0
        self.wb_set = False
        self.rh_set = False
        self.hr_set = False
        self.dp_set = False
        self.h_set = False
        self.rho_set = False
        self.C_p = fr_u(1.006, "kJ/kg/K")
        if len(kwargs) > 1:
            raise RuntimeError(
                f"{PsychState.__name__} can only be initialized with a single key word argument, but received {len(kwargs)}: {kwargs}"
            )
        if "wetbulb" in kwargs:
            self.wb = kwargs["wetbulb"]
        elif "hum_rat" in kwargs:
            self.hr = kwargs["hum_rat"]
        elif "rel_hum" in kwargs:
            self.rh = kwargs["rel_hum"]
        elif "enthalpy" in kwargs:
            self.h = kwargs["enthalpy"]
        else:
            raise RuntimeError(f"{PsychState.__name__}: Unknown or missing key word argument {kwargs}.")

    @property
    def wb(self):
        if self.wb_set:
            return self._wb
        raise RuntimeError("Wetbulb not set")

    @wb.setter
    def wb(self, wb):
        self._wb = wb
        self.wb_C = to_u(self._wb, "°C")
        self.wb_set = True

    def get_wb_C(self):
        if self.wb_set:
            return self.wb_C
        else:
            raise RuntimeError("Wetbulb not set")

    @property
    def hr(self):
        if self.hr_set:
            return self._hr
        else:
            self.hr = psychrolib.GetHumRatioFromTWetBulb(self.db_C, self.get_wb_C(), self.p)
            return self._hr

    @hr.setter
    def hr(self, hr):
        self._hr = hr
        if not self.wb_set:
            self.wb = fr_u(
                psychrolib.GetTWetBulbFromHumRatio(self.db_C, self._hr, self.p),
                "°C",
            )

        self.hr_set = True

    @property
    def rh(self):
        if self.rh_set:
            return self._rh
        else:
            self.rh = psychrolib.GetHumRatioFromTWetBulb(self.db_C, self.get_wb_C(), self.p)
            return self._rh

    @rh.setter
    def rh(self, rh):
        self._rh = rh
        if not self.wb_set:
            self.wb = fr_u(psychrolib.GetTWetBulbFromRelHum(self.db_C, self._rh, self.p), "°C")
        self.rh_set = True

    @property
    def h(self):
        if self.h_set:
            return self._h
        else:
            self.h = psychrolib.GetMoistAirEnthalpy(self.db_C, self.hr)
            return self._h

    @h.setter
    def h(self, h):
        self._h = h
        if not self.hr_set:
            self.hr = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(self._h, self.db_C)
        self.h_set = True

    @property
    def rho(self):
        if self.rho_set:
            return self._rho
        else:
            self.rho = psychrolib.GetMoistAirDensity(self.db_C, self.hr, self.p)
            return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.rho_set = True


STANDARD_CONDITIONS = PsychState(drybulb=fr_u(70.0, "°F"), hum_rat=0.0)


def cooling_psych_state(drybulb=fr_u(95.0, "°F"), pressure=fr_u(1.0, "atm")):
    "Applies default assumption for outdoor cooling humidity conditions in AHRI Standards"
    return PsychState(drybulb=drybulb, rel_hum=0.4, pressure=pressure)


def heating_psych_state(drybulb=fr_u(95.0, "°F"), pressure=fr_u(1.0, "atm")):
    "Applies default assumption for outdoor heating humidity conditions in AHRI Standards"
    return PsychState(drybulb=drybulb, wetbulb=drybulb - fr_u(2.0, "delta_degF"), pressure=pressure)

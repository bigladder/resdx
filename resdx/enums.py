from enum import Enum


class FanMotorType(Enum):
    UNKNOWN = 0
    PSC = 1
    BPM = 2


class StagingType(Enum):
    SINGLE_STAGE = 1
    TWO_STAGE = 2
    VARIABLE_SPEED = 3

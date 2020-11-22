from enum import Enum, unique


@unique
class Action(Enum):
    level_0 = 1.0
    level_1 = 0.8
    level_2 = 0.6
    level_3 = 0.4
    level_4 = 0.2
    level_5 = 0.0

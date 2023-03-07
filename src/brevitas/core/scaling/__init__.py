from brevitas.inject.enum import ScalingImplType

assert ScalingImplType

from brevitas.core.stats import SCALAR_SHAPE

from .int_scaling import IntScaling
from .int_scaling import PowerOfTwoIntScaling
from .runtime import RuntimeStatsScaling
from .runtime import StatsFromParameterScaling
from .standalone import ConstScaling
from .standalone import ParameterFromRuntimeStatsScaling
from .standalone import ParameterScaling
from .standalone import AccumulatorAwareParameterScaling

SCALING_STATS_REDUCE_DIM = 1

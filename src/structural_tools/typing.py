from typing import TypeAlias

import numpy as np
from forallpeople import Physical

FloatLike: TypeAlias = float | int | Physical | np.number

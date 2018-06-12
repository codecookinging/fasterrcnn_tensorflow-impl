# -*- coding: utf-8 -*-
from lib.model.config import cfg
from lib.roi_data_layer.minibatch import get_minibatch
import numpy as np
import time


st0 = np.random.get_state()
millis = int(round(time.time() * 1000)) % 4294967295
np.random.seed(millis)

print(st0)
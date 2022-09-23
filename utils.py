import os
import random
import torch
import numpy as np

def set_seed(manualSeed=0):
   random.seed(manualSeed)
   np.random.seed(manualSeed)
   torch.manual_seed(manualSeed)
   torch.cuda.manual_seed(manualSeed)
   torch.cuda.manual_seed_all(manualSeed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   os.environ['PYTHONHASHSEED'] = str(manualSeed)
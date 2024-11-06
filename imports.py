import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from flax import linen as nn
from jax_sysid.models import Model
from jax_sysid.models import StaticModel
from jax_sysid.utils import compute_scores, standard_scale, unscale
import casadi as ca
import matplotlib.pyplot as plt
from pycvxset import Polytope
from functools import partial
from pathlib import Path
from itertools import combinations
import pickle
import time


current_directory = Path(__file__).parent


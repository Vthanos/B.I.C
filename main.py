import tensorflow as tf
import PIL
import numpy
import matplotlib as mplt
import pandas as pd
import os

for dirname, _, filenames in os.walk('inputs/jpeg'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

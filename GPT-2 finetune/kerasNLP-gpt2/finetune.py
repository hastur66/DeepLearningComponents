import numpy as np
import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow import keras
import time

start = time.time()

# load cnn_dailymail dataset
cnn_ds = tfds.load("cnn_dailymail", as_supervised=True)

end = time.time()
print("Dataset Load Time:", end - start)


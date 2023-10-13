import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tf_text
from tensorflow import keras
import numpy as np

import time
import progressbar

from nltk import tokenize
import nltk


nltk.download('punkt')

start = time.time()

# load cnn_dailymail dataset
cnn_ds = tfds.load("cnn_dailymail", as_supervised=True)

end = time.time()
print("Dataset Load Time:", end - start)

def view_data():
    for article, highlights in cnn_ds['train']:
        print(article.numpy())
        print(highlights.numpy())
        break

view_data()

# load model, tokenizer, and preprocessor
gpt2_tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset("gpt2_base_en")
gpt2_preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_base_en",
    sequence_length=256,
    add_end_token=True,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=gpt2_preprocessor)
     

def merge_sentences(sentence, max_length):
    res = []
    cur_len = 0
    cur_sentence = []
    for s in sentence:
        if cur_len + len(s) > max_length:
            res.append(" ".join(cur_sentence))
            cur_len = len(s)
            cur_sentence = [s]
        else:
            cur_len += len(s)
            cur_sentence.append(s)
    res.append(" ".join(cur_sentence))
    return res

max_length = 512
all_sentences = []
count = 0
total = len(cnn_ds["train"])
num_articles_to_process = 20000
progressbar_update_freq = 2000

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]

# Render a progressbar to track progress
bar = progressbar.ProgressBar(
    max_value=num_articles_to_process // progressbar_update_freq + 2,
    widgets=widgets).start()

for article, highlight in cnn_ds['train']:
  # Use NLTK tokenize to split articles into sentences
  sentences = tokenize.sent_tokenize(str(article))
  # Merge individual sentences into longer context
  combined_res = merge_sentences(sentences, max_length)
  # Add merged context into collection
  all_sentences.extend(combined_res)
  count += 1
  if count % progressbar_update_freq == 0:
    bar.update(count / progressbar_update_freq)
  if count >= num_articles_to_process:
    break

tf_train_ds = tf.data.Dataset.from_tensor_slices(all_sentences)
processed_ds = tf_train_ds.map(gpt2_preprocessor, tf.data.AUTOTUNE)
part_of_ds = processed_ds.take(100)

gpt2_lm.include_preprocessing = False

num_epochs = 1

lr = tf.keras.optimizers.schedules.PolynomialDecay(
    5e-5,
    decay_steps=part_of_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(
    optimizer=keras.optimizers.experimental.Adam(lr),
    loss=loss,
    weighted_metrics=["accuracy"])

gpt2_lm.fit(part_of_ds, epochs=num_epochs)
     

start = time.time()

output = gpt2_lm.generate("Breaking news: the river", max_length=200)
print("\nGPT-2 output:")
print(output.numpy().decode("utf-8"))

end = time.time()
print("TOTAL TIME ELAPSED: ", end - start)

del gpt2_tokenizer, gpt2_preprocessor, gpt2_lm
import keras_nlp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow import keras
from tensorflow.lite.python import interpreter
import time

gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en_cnn_dailymail")

@tf.function
def generate(prompt, max_length):
    return gpt2_lm.generate(prompt, max_length)

concrete_func = generate.get_concrete_function(tf.TensorSpec([], tf.string), 100)

def run_inference(input, generate_tflite):
    interp = interpreter.InterpreterWithCustomOps(
        model_content=generate_tflite,
        custom_op_registerers=tf_text.tflite_registrar.SELECT_TFTEXT_OPS
    )
    interp.get_signature_list()

    generator = interp.get_signature_runner("serving_default")
    output = generator(prompt=np.array([input]))
    print("\nGenerated with TFLite:\n", output["output_0"])
    

gpt2_lm.jit_compile = False

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], gpt2_lm)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.experimental_select_user_tf_ops = ["UnsortedSegmentJoin", "UpperBound"]
converter._experimental_guarantee_all_funcs_one_use = True
quant_generate_tflite = converter.convert()

with open('unquantized_gpt2.tflite', 'wb') as f:
  f.write(quant_generate_tflite)
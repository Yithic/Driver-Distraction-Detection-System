import tensorflow as tf
from common import run_live_inference

print("Loading Custom 2D CNN...")
model = tf.keras.models.load_model("Custom_CNN_SFDD_Optimized.keras")
run_live_inference(model, window_name="Custom 2D CNN Monitor")
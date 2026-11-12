import tensorflow as tf
from common import run_live_inference

print("Loading ResNet50...")
model = tf.keras.models.load_model("ResNet50_SFDD_Optimized.keras")
run_live_inference(model, window_name="ResNet50 Monitor")
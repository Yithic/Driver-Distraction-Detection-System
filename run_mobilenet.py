import tensorflow as tf
from common import run_live_inference

print("Loading MobileNetV2...")
model = tf.keras.models.load_model("MobileNetV2_SFDD_Optimized.keras")
run_live_inference(model, window_name="MobileNetV2 Monitor")
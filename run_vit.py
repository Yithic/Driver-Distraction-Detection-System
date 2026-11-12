import tensorflow as tf
from common import run_live_inference

print("Loading Vision Transformer (ViT)...")
# safe_mode=False is required due to TFSMLayer loading
model = tf.keras.models.load_model("ViT_SFDD.keras", safe_mode=False)
run_live_inference(model, window_name="Vision Transformer Monitor")
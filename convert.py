import os
import tensorflow as tf

def convert_to_tflite():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # 1. Convert CNN model
    cnn_keras_path = os.path.join(models_dir, "cnn_garbage_best.keras")
    cnn_tflite_path = os.path.join(models_dir, "cnn_garbage_best.tflite")
    if os.path.exists(cnn_keras_path):
        print(f"Loading Keras model from {cnn_keras_path}...")
        model = tf.keras.models.load_model(cnn_keras_path)
        print("Converting CNN to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(cnn_tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved CNN TFLite model to {cnn_tflite_path} ({os.path.getsize(cnn_tflite_path) / 1024 / 1024:.2f} MB)")
    else:
        print(f"CNN Keras model not found at {cnn_keras_path}")

    # 2. Convert MobileNetV2 model
    mobilenet_keras_path = os.path.join(models_dir, "mobilenetv2_garbage_best.keras")
    mobilenet_tflite_path = os.path.join(models_dir, "mobilenetv2_garbage_best.tflite")
    if os.path.exists(mobilenet_keras_path):
        print(f"Loading Keras model from {mobilenet_keras_path}...")
        model = tf.keras.models.load_model(mobilenet_keras_path)
        print("Converting MobileNetV2 to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(mobilenet_tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved MobileNetV2 TFLite model to {mobilenet_tflite_path} ({os.path.getsize(mobilenet_tflite_path) / 1024 / 1024:.2f} MB)")
    else:
        print(f"MobileNetV2 Keras model not found at {mobilenet_keras_path}")

if __name__ == "__main__":
    convert_to_tflite()

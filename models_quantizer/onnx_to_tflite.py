import tensorflow as tf
import numpy as np
import os
import subprocess

def representative_dataset_gen():
    """Generate representative dataset for quantization"""
    # Load a sample image for calibration
    from PIL import Image
    image = Image.open('test_image.jpg').convert('RGB')
    image = image.resize((256, 256))
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Generate 100 samples with slight variations
    for _ in range(100):
        # Add random noise
        noise = np.random.normal(0, 0.01, img_array.shape)
        sample = np.clip(img_array + noise, 0, 1)
        # Reshape to match model input
        sample = sample.transpose(2, 0, 1)  # HWC to CHW
        sample = np.expand_dims(sample, 0)  # Add batch dimension
        yield [sample]

def convert_onnx_to_tflite():
    """Convert ONNX model to TFLite format using onnx2tf"""
    print("Starting ONNX to TFLite conversion...")
    
    # Input and output paths
    onnx_path = 'realesrgan_x4_quantized.onnx'
    output_dir = 'saved_model'
    
    # Convert ONNX to TensorFlow SavedModel using onnx2tf
    # -i: input ONNX file
    # -ois: override input shape to static values (batch_size=1, channels=3, height=64, width=64)
    # -v info: set verbosity to info level
    cmd = [
        'onnx2tf',
        '-i', onnx_path,
        '-ois', 'input:1,3,64,64',
        '-v', 'info'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Model converted to TensorFlow format: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting ONNX to TensorFlow: {e}")
        return
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_path = 'realesrgan_x4.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted to TFLite: {tflite_path}")
    
    # Create quantized version
    print("\nCreating quantized TFLite model...")
    quant_converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Add quantization parameters
    quant_converter.representative_dataset = representative_dataset_gen
    quant_converter.target_spec.supported_types = [tf.int8]
    quant_converter.inference_input_type = tf.uint8
    quant_converter.inference_output_type = tf.uint8
    
    # Convert the quantized model
    tflite_quant_model = quant_converter.convert()
    
    # Save the quantized TFLite model
    quant_tflite_path = 'realesrgan_x4_quantized.tflite'
    with open(quant_tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"Quantized TFLite model saved as: {quant_tflite_path}")

if __name__ == "__main__":
    convert_onnx_to_tflite() 
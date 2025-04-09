import torch
import onnx
import numpy as np
from PIL import Image
from RealESRGAN import RealESRGAN
import onnxruntime
import time
import tensorflow as tf

def preprocess_image(image_path, input_size=256):
    """Preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    # Resize to a larger size for better quality
    image = image.resize((input_size, input_size))
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # Transpose to CHW format
    img_array = img_array.transpose(2, 0, 1)
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array, image

def postprocess_output(output):
    """Convert model output to image"""
    # Remove batch dimension and transpose back to HWC
    output = output[0].transpose(1, 2, 0)
    # Clip values and convert to uint8
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

def test_original_model(image_path):
    """Test the original PyTorch model"""
    print("\nTesting original PyTorch model...")
    
    # Load model
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('RealESRGAN_x4.pth')
    model.model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    # Time the inference
    start_time = time.time()
    sr_image = model.predict(image)
    inference_time = time.time() - start_time
    
    # Save result
    sr_image.save('sr_image_original.png')
    print(f"Original model inference time: {inference_time:.2f} seconds")
    print("Result saved as sr_image_original.png")

def test_quantized_model(image_path):
    """Test the quantized ONNX model"""
    print("\nTesting quantized ONNX model...")
    
    # Load the ONNX model
    session = onnxruntime.InferenceSession('realesrgan_x4_quantized.onnx')
    
    # Preprocess image with larger input size (256x256)
    input_data, original_image = preprocess_image(image_path, input_size=256)
    
    # Time the inference
    start_time = time.time()
    output = session.run(['output'], {'input': input_data})[0]
    inference_time = time.time() - start_time
    
    # Postprocess and save result
    sr_image = postprocess_output(output)
    sr_image.save('sr_image_quantized.png')
    print(f"Quantized model inference time: {inference_time:.2f} seconds")
    print("Result saved as sr_image_quantized.png")

def test_tflite_model(image_path):
    """Test TFLite model"""
    print("\nTesting TFLite model...")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_size = 256  # Use larger input size for better quality
    image = image.resize((input_size, input_size))
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Reshape to match model input (NCHW format)
    input_data = img_array.transpose(2, 0, 1)  # HWC to CHW
    input_data = np.expand_dims(input_data, 0)  # Add batch dimension
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path='realesrgan_x4.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    print(f"TFLite model inference time: {inference_time:.2f} seconds")
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Post-process output
    output_data = output_data[0].transpose(1, 2, 0)  # CHW to HWC
    output_data = np.clip(output_data, 0, 1)
    output_data = (output_data * 255).astype(np.uint8)
    
    # Save result
    output_image = Image.fromarray(output_data)
    output_image.save('sr_image_tflite.png')
    print("Result saved as sr_image_tflite.png")

def test_quantized_tflite_model(image_path):
    """Test quantized TFLite model"""
    print("\nTesting quantized TFLite model...")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_size = 256  # Use larger input size for better quality
    image = image.resize((input_size, input_size))
    img_array = np.array(image).astype(np.uint8)  # Keep as uint8 for quantized model
    
    # Reshape to match model input (NCHW format)
    input_data = img_array.transpose(2, 0, 1)  # HWC to CHW
    input_data = np.expand_dims(input_data, 0)  # Add batch dimension
    
    # Load quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path='realesrgan_x4_quantized.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time
    print(f"Quantized TFLite model inference time: {inference_time:.2f} seconds")
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Post-process output
    output_data = output_data[0].transpose(1, 2, 0)  # CHW to HWC
    output_data = output_data.astype(np.uint8)
    
    # Save result
    output_image = Image.fromarray(output_data)
    output_image.save('sr_image_tflite_quantized.png')
    print("Result saved as sr_image_tflite_quantized.png")

if __name__ == "__main__":
    image_path = 'test_image.jpg'
    
    # Test all models
    #test_original_model(image_path)
    test_quantized_model(image_path)
    #test_tflite_model(image_path)
    #test_quantized_tflite_model(image_path)
    
    print("\nTesting complete! Check all output images for results.") 
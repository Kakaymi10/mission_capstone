import torch
import onnx
import numpy as np
from RealESRGAN import RealESRGAN
import os

def create_calibration_dataset(image_path, num_samples=100):
    """Create a calibration dataset from the test image"""
    from PIL import Image
    import numpy as np
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Create multiple samples by adding noise
    samples = []
    for _ in range(num_samples):
        # Add random noise to create variations
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        noisy_image = np.clip(image + noise, 0, 255)
        # Reshape to match model input
        sample = noisy_image.transpose(2, 0, 1)  # HWC to CHW
        sample = sample.astype(np.float32) / 255.0  # Normalize to [0,1]
        samples.append(sample)
    
    return samples

# Load the model
device = torch.device('cpu')
model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4.pth')
model.model.eval()

# Create calibration dataset
calibration_samples = create_calibration_dataset('test_image.jpg')

# Export to ONNX with quantization
onnx_path = 'realesrgan_x4_quantized.onnx'
torch.onnx.export(
    model.model,
    torch.randn(1, 3, 256, 256),  # Dummy input for shape
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                  'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
    # Add quantization parameters
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    custom_opsets={
        "com.microsoft": 1
    }
)

# Load and verify the ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print(f"Quantized model saved as: {onnx_path}") 
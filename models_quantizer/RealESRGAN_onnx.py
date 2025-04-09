import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('RealESRGAN_x4.pth')

path_to_image = 'test_image.jpg'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('sr_image.png')

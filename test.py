import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from network.Transformer import Transformer

# Argument parser to accept command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='test_img', help="Directory with input images")
parser.add_argument('--load_size', type=int, default=450, help="Resize size of images")
parser.add_argument('--model_path', default='./pretrained_model', help="Path to pre-trained model")
parser.add_argument('--style', default='Hayao', help="Style for the model (e.g., Miyazaki, Hayao)")
parser.add_argument('--output_dir', default='test_output', help="Directory to save output images")
parser.add_argument('--gpu', type=int, default=0, help="GPU device ID, use -1 for CPU")

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

# Create output directory if it does not exist
if not os.path.exists(opt.output_dir):
    os.mkdir(opt.output_dir)

# Device selection: Check if GPU is available and user chose to use it
device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu >= 0 else "cpu")

# Load pre-trained model
model = Transformer()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + '_net_G_float.pth'), weights_only=True))  # Avoid the deprecation warning
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move model to the selected device (GPU or CPU)

print(f"Running on {device} mode")

# Process each image in the input directory
for files in os.listdir(opt.input_dir):
    ext = os.path.splitext(files)[1]
    
    # Skip files that are not .jpg or .png
    if ext not in valid_ext:
        continue

    # Load image
    input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
    
    # Resize image while maintaining aspect ratio
    w, h = input_image.size  # Unpack size (width, height)
    
    # Ensure w and h are integers (safety)
    w = int(w)
    h = int(h)
    
    # Maintain aspect ratio during resizing
    ratio = h / float(w)  # Ratio of height to width
    if ratio > 1:  # If the height is larger than the width
        h = opt.load_size
        w = int(h / ratio)  # Calculate width to maintain the aspect ratio
    else:  # If width is larger than height
        w = opt.load_size
        h = int(w * ratio)  # Calculate height to maintain the aspect ratio
    
    # Resize the image
    input_image = input_image.resize((w, h), Image.BICUBIC)
    input_image = np.asarray(input_image)
    
    # Convert RGB -> BGR (which is often required for models trained on certain datasets)
    input_image = input_image[:, :, [2, 1, 0]]
    
    # Transform to tensor
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    
    # Preprocess: Normalize to [-1, 1]
    input_image = -1 + 2 * input_image
    
    # Move input image to the selected device (GPU or CPU)
    input_image = input_image.to(device)

    # Forward pass through the model
    with torch.no_grad():  # No need to track gradients during inference
        output_image = model(input_image)
        output_image = output_image[0]  # Get the first image from the batch

    # Convert BGR -> RGB for output
    output_image = output_image[[2, 1, 0], :, :]  # Convert channels back to RGB

    # Deprocess output: Convert back from [-1, 1] to [0, 1]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    
    # Save the output image
    output_filename = os.path.join(opt.output_dir, files[:-4] + '_' + opt.style + '.jpg')  # Handle file extension
    vutils.save_image(output_image, output_filename)

print('Done!')
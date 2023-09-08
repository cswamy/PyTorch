"""
Contains utility functions:
1. download_image_data to download and unzip data
2. save_model saves model state_dict
3. load_model loads a saved model's state_dict
4. predict_image classifies a custom image using trained model
"""

# Import packages
import requests
import zipfile
from pathlib import Path
from typing import List

import torch
import torchvision

def download_image_data(target_folder: str,
                        data_source: str):

  """
  Downloads and unzips image data from data source into specified folder.

  Args:
  target_folder (str): provide a parent folder name for data.
  data_source (str): web link containing data source files. For example, 
  this could be a github link.

  Returns:
  image_path (POSIX path): path to directory where data is downloaded
  and unzipped.
  """

  # Setup path to data folder
  data_path = Path("data/")
  image_path = data_path / target_folder

  # Create folder for data if not exists
  if image_path.is_dir():
    print(f"{image_path} exists already.")
  else:
    print(f"Creating {image_path} directory...")
    try:
      image_path.mkdir(parents=True, exist_ok=True)
    except:
      print(f"Failed to create {image_path}. Exiting!")
      return

  # Download or overwrite data into folder
  zip_file_name = target_folder + ".zip"
  with open(file=data_path/zip_file_name, mode="wb") as f:
    try:
      request = requests.get(data_source)
      print("Downloading data...")
      f.write(request.content)
    except:
      print("Failed to download data. Exiting!")
      return

  # Unzip data
  with zipfile.ZipFile(file=data_path/zip_file_name, mode="r") as zip_ref:
    try:
      print("Unzipping file...")
      zip_ref.extractall(image_path)
    except:
      print("Failed to unzip data. Exiting!")
      return

  print("Data downloaded and unzipped succesfully!")
  # Delete zip file
  zip_file = data_path/zip_file_name
  zip_file.unlink()

  # Return image path if everything was successfully executed
  return image_path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  """
  Save a PyTorch model to specified directory with specified name
  Args:
    model(torch.nn.Module): A trained PyTorch model.
    target_dir(str): Target directory to save model.
    model_name(str): Model name to save model.
  Returns:
    None. Prints to screen when save is successful.
  """

  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def load_model(model: torch.nn.Module,
               model_path: str):
  """
  Loads a PyTorch model's state_dict from given file path into a new model instance.
  Args:
    model(torch.nn.Module): New instance of a PyTorch model defined with a model building class.
    model_path(str): Path to a previously trained and saved PyTorch model.
  Returns:
    None. Loads pre-trained saved model's state dict to new instance.
  """
  print(f"[INFO] Loading model from {model_path}")
  model.load_state_dict(torch.load(model_path))

def predict_image(model: torch.nn.Module,
                  image_path: str,
                  class_names: List,
                  device: torch.device):
  """
  Classifies a custom image using an image classification model.
  Args:
    model(torch.nn.Module): PyTorch model instance loaded with parameters of a pre-trained model.
    image_path(str): Path to a custom image for classification.
    class_names(List): List of class names for classification.
  Returns:
    None. Prints image class to screen.
  """

  # Read image from path and turn into torch.float32 tensor
  image = torchvision.io.read_image(image_path).type(torch.float32)

  # Scale values of tensor to values between 0 and 1
  image = image / 255

  # Resize image to be same size as training data
  transform = torchvision.transforms.Resize(size=(64, 64))
  image = transform(image)

  # Add a batch dimension similar to training data
  image = image.unsqueeze(dim=0)

  # Predict on image
  model.eval()
  with torch.inference_mode():
    # Send image to device
    image = image.to(device)

    # Forward pass
    pred_logit = model(image)
    pred_label = torch.softmax(pred_logit, dim=1).argmax(dim=1)
    pred_class = class_names[pred_label]

  print(f"Predicted class for image is {pred_class}")

"""
Contains pytorch utility functions:
1. save_model saves model state_dict
2. load_model loads a saved model's state_dict
"""

import torch
import torchvision

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

import torch
from typing import Any
import os

from .renderer import Renderer
from .mesh_renderer import MeshRenderer
from .skeleton_renderer import SkeletonRenderer
from .pose_utils import eval_pose, Evaluator

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x

def get_parent_folder_of_package(package_name):
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = os.path.abspath(package.__file__)

    # Get the directory of the package
    package_dir = os.path.dirname(package_path)

    # Get the parent directory
    parent_dir = os.path.dirname(package_dir)

    return parent_dir
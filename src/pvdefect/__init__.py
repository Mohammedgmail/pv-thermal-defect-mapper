"""Find hot spots in drone thermal images and map each one to the exact PV module and cell,
using the camera poses and plant model of a pv-drone-inspection-planner mission."""

from .mapper import Finding, Mapper, Settings, load_survey
from .refine import refine_poses
from .scene import Module, Pose, Scene

__version__ = "0.1.0"
__all__ = [
    "Finding",
    "Mapper",
    "Module",
    "Pose",
    "Scene",
    "Settings",
    "load_survey",
    "refine_poses",
]

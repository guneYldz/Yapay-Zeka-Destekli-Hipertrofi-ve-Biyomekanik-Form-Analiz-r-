"""
Pure geometry module for domain logic.
MANDATORY PURE: Only built-in math module allowed.
"""

import math
from typing import Optional

from src.domain.entities import Point3D


def calculate_angle_3d(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """
    Calculate the 3D angle formed by p1 - p2 - p3.
    p2 is the vertex of the angle.
    Returns angle in degrees [0, 180].
    """
    # Vectors
    v1_x = p1.x - p2.x
    v1_y = p1.y - p2.y
    v1_z = p1.z - p2.z
    
    v2_x = p3.x - p2.x
    v2_y = p3.y - p2.y
    v2_z = p3.z - p2.z
    
    # Dot product
    dot_product = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z
    
    # Magnitudes
    mag_v1 = math.sqrt(v1_x**2 + v1_y**2 + v1_z**2)
    mag_v2 = math.sqrt(v2_x**2 + v2_y**2 + v2_z**2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0
        
    cos_angle = dot_product / (mag_v1 * mag_v2)
    # Clamp to [-1, 1] to avoid float precision issues
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def calculate_angle_2d(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """
    Calculate the 2D angle (XY plane) formed by p1 - p2 - p3.
    p2 is the vertex of the angle.
    Returns absolute angle in degrees [0, 180].
    """
    # Using arctan2
    radians = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    angle = abs(math.degrees(radians))
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle


def calculate_vertical_angle(p1: Point3D, p2: Point3D) -> float:
    """
    Calculate the deviation from the vertical (Y axis) for vector p1 -> p2.
    Useful for measuring back leaning angle.
    Returns angle in degrees [0, 90].
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # math.atan2(x, y) gives angle with Y axis directly
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
    return angle

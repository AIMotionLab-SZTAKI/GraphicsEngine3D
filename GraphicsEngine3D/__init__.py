import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parents[1]))

from GraphicsEngine3D.main import GraphicsEngine

__all__ = ['GraphicsEngine']
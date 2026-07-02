# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..mouse_data import MouseData



# ================================================================
# 1. Section: Classes
# ================================================================
@dataclass
class AbstractTransformation(ABC):
    @abstractmethod
    def apply(self, mouse: MouseData) -> MouseData:
        raise NotImplementedError("apply method must be implemented by subclass")

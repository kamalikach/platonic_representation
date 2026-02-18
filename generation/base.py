from abc import ABC, abstractmethod
from pathlib import Path


class DataGenerator(ABC):
    """Abstract base class for dataset generation methods."""

    @abstractmethod
    def generate(self, cfg) -> None:
        """
        Generate a dataset and save it to cfg.output_dir.
        
        Args:
            cfg: Hydra config object
        """
        pass


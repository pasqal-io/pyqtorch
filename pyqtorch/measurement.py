from utils import MeasurementMode

class Measurements:
    """Measurements handling class.

    Attributes:
        protocol: Measurement protocol applied.
        options: Dictionary of options.
    """
    def __init__(self, protocol: MeasurementMode, options: dict) -> None:
        self.protocol = protocol
        self.options = options
    
        self._generator_map: dict = {
            MeasurementMode.TOMOGRAPHY: self._tomography_expectation,
        }
    
    def _tomography_expectation(self):
        raise NotImplementedError
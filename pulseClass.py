import numpy as np
import matplotlib.pyplot as plt
from types import FunctionType

class Pulse:
    def __init__(self):
        self.parameters = {
            'Omega': {'func': lambda t, p: 0.0, 'type': 'constant'},
            'phi': {'func': lambda t, p: 0.0, 'type': 'constant'},
            'delta': {'func': lambda t, p: 0.0, 'type': 'constant'}
        }

    def set_parameter(self, param: str, shape: str, **kwargs):
        """Set parameter waveform (Omega, phi, or delta)"""
        if param not in ['Omega', 'phi', 'delta']:
            raise ValueError("Invalid parameter. Choose 'Omega', 'phi', or 'delta'")

        if shape == 'constant':
            self.parameters[param] = {
                'func': self._constant_shape(kwargs.get('value', 0.0)),
                'type': 'constant'
            }
        elif shape == 'gaussian':
            self.parameters[param] = {
                'func': self._gaussian_shape(
                    kwargs['amp'], 
                    kwargs['t0'], 
                    kwargs['sigma']
                ),
                'type': 'gaussian'
            }
        elif shape == 'linear':
            self.parameters[param] = {
                'func': self._linear_shape(
                    kwargs['slope'],
                    kwargs.get('intercept', 0.0)
                ),
                'type': 'linear'
            }
        elif shape == 'custom':
            if not isinstance(kwargs['func'], FunctionType):
                raise ValueError("Custom shape requires a function f(t, params)")
            self.parameters[param] = {
                'func': kwargs['func'],
                'type': 'custom'
            }
        else:
            raise ValueError(f"Unknown shape type: {shape}")

    @staticmethod
    def _constant_shape(value: float):
        return lambda t, p: value

    @staticmethod
    def _gaussian_shape(amp: float, t0: float, sigma: float):
        return lambda t, p: amp * np.exp(-(t - t0)**2 / (2*sigma**2))

    @staticmethod
    def _linear_shape(slope: float, intercept: float):
        return lambda t, p: slope * t + intercept

    def Omega(self, t: float) -> float:
        return self.parameters['Omega']['func'](t, self.parameters)

    def phi(self, t: float) -> float:
        return self.parameters['phi']['func'](t, self.parameters)

    def delta(self, t: float) -> float:
        return self.parameters['delta']['func'](t, self.parameters)

    def plot(self, t_range=(0, 5), num_points=200):
        """Visualize all pulse parameters"""
        t = np.linspace(*t_range, num_points)
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        axs[0].plot(t, [self.Omega(τ) for τ in t], 'C0')
        axs[0].set_ylabel('Ω(t) [MHz]')
        
        axs[1].plot(t, [self.phi(τ) for τ in t], 'C1')
        axs[1].set_ylabel('ϕ(t) [rad]')
        
        axs[2].plot(t, [self.delta(τ) for τ in t], 'C2')
        axs[2].set_ylabel('δ(t) [MHz]')
        axs[2].set_xlabel('Time [µs]')
        
        plt.suptitle("Pulse Parameters")
        plt.tight_layout()
        plt.show()
import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize the 1 Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing at low speeds.
            beta: Speed coefficient. Higher = less lag at high speeds.
            d_cutoff: Cutoff frequency for derivative (Hz). Usually 1.0.
        """
        self.first_time = True
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None
        
    def __call__(self, x, t=None):
        """
        Filter the signal.
        
        Args:
            x: Noisy input value (float or numpy array)
            t: Timestamp (seconds). If None, uses time.time().
            
        Returns:
            Filtered value.
        """
        if t is None:
            t = time.time()
            
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0 * x  # Zero of same shape/type
            self._t_prev = t
            return x
            
        # Compute time step
        dt = t - self._t_prev
        self._t_prev = t
        
        # Avoid division by zero
        if dt <= 0:
            return self._x_prev
            
        # Compute derivative (velocity)
        alpha_d = self._smoothing_factor(dt, self._d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = self._exponential_smoothing(alpha_d, dx, self._dx_prev)
        
        # Compute cutoff frequency based on speed
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        
        # Filter signal
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = self._exponential_smoothing(alpha, x, self._x_prev)
        
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        
        return x_hat
        
    def _smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)
        
    def _exponential_smoothing(self, alpha, x, x_prev):
        return alpha * x + (1 - alpha) * x_prev

class ARPoseFilter:
    """Wrapper for filtering Translation and Rotation vectors."""
    def __init__(self, min_cutoff=0.1, beta=0.01, d_cutoff=1.0):
        # Separate filters for T (Translation) and R (Rotation)
        self.t_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.r_filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        
    def filter(self, rvec, tvec):
        """Apply 1 Euro Filter to pose vectors."""
        # T-vec is straightforward (Euclidean space)
        filtered_t = self.t_filter(tvec)
        
        # R-vec (Rodrigues) is roughly Euclidean for small changes
        # For rigorous production, we'd use Quaternions, but 1€ on rvec is 
        # already vastly superior to simple Lerp.
        filtered_r = self.r_filter(rvec)
        
        return filtered_r, filtered_t
    
    def reset(self):
        self.t_filter = OneEuroFilter(self.t_filter._min_cutoff, self.t_filter._beta, self.t_filter._d_cutoff)
        self.r_filter = OneEuroFilter(self.r_filter._min_cutoff, self.r_filter._beta, self.r_filter._d_cutoff)

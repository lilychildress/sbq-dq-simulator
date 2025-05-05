
import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows

from bff_paper_figures.inner_product_functions import InnerProductSettings

# Take the inner product of the signal with a complex exponential
def inner_product_exponent(
    inner_product_freq_hz: float,
    sample_times_s: NDArray,
    dq_signal: NDArray,
    axis: int = 0,
) -> float:
    numerator = np.sum(np.exp(-1.0j* 2 * np.pi * (inner_product_freq_hz) * sample_times_s) * dq_signal, axis=axis)
    denominator = np.sum(
        np.exp(-1.0j*2 * np.pi * (inner_product_freq_hz) * sample_times_s)
        * np.exp(1.0j*2 * np.pi * (inner_product_freq_hz) * sample_times_s)
    )
    return numerator / denominator

def double_inner_product_exponential(
    dq_signal: NDArray, rabi_freq_hz: float, ramsey_freq_hz: float, ips: InnerProductSettings, 
):
    if ips.use_effective_rabi_frequency:
        rabi_freq_hz = np.sqrt(rabi_freq_hz**2 + ramsey_freq_hz**2)

    rabi_window = windows.get_window(ips.rabi_window, len(ips.mw_pulse_durations_s))

    # Perform the inner product along the pulse-duration dimension (leaves a time-domain Ramsey signal)
    rabi_inner_product = inner_product_exponent(
        rabi_freq_hz, ips.mw_pulse_durations_s, rabi_window * np.transpose(dq_signal), axis=1
    )

    if ips.subtract_mean:
        rabi_inner_product = rabi_inner_product - np.mean(rabi_inner_product)

    # Perform the inner product along the evolution-time dimension
    ramsey_window = windows.get_window(ips.ramsey_window, len(ips.free_evolution_times_s))
    double_inner_product_exp_exp = inner_product_exponent(
        ramsey_freq_hz, ips.free_evolution_times_s, ramsey_window * rabi_inner_product, axis=0
    )

    return double_inner_product_exp_exp

def inner_product_exp_to_minimize(ramsey_rabi_hz, truncated_signal, inner_product_settings):
    ramsey_hz,rabi_hz = ramsey_rabi_hz
    return -np.abs(double_inner_product_exponential(truncated_signal, rabi_hz, ramsey_hz, inner_product_settings))

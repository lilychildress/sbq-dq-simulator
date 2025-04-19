from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows


@dataclass
class InnerProductSettings:  # Settings shared by all inner products performed in a given inversion protocol
    mw_pulse_durations_s: NDArray
    free_evolution_times_s: NDArray
    rabi_window: str = "boxcar"  # Best option: "blackman"
    ramsey_window: str = "boxcar"  # Best option: "boxcar"
    subtract_mean: bool = True  # Best option: True
    use_effective_rabi_frequency: bool = True  # Best option: True (for frequency domain inversion)


# Take the inner product of the signal with a sinusoidal function (either np.cos or np.sin)
def inner_product_sinusoid(
    sinusoid_function: np.ufunc,
    inner_product_freq_hz: float,
    sample_times_s: NDArray,
    dq_signal: NDArray,
    axis: int = 0,
) -> float:
    numerator = np.sum(sinusoid_function(2 * np.pi * (inner_product_freq_hz) * sample_times_s) * dq_signal, axis=axis)
    denominator = np.sum(
        sinusoid_function(2 * np.pi * (inner_product_freq_hz) * sample_times_s)
        * sinusoid_function(2 * np.pi * (inner_product_freq_hz) * sample_times_s)
    )
    return numerator / denominator


# Calculates the inner product I(rabi_freq, ramsey_freq) for specified values of rabi_freq and ramsey_freq
def double_cosine_inner_product(
    dq_signal: NDArray, rabi_freq_hz: float, ramsey_freq_hz: float, ips: InnerProductSettings
):
    if ips.use_effective_rabi_frequency:
        rabi_freq_hz = np.sqrt(rabi_freq_hz**2 + ramsey_freq_hz**2)

    rabi_window = windows.get_window(ips.rabi_window, len(ips.mw_pulse_durations_s))

    # Perform the inner product along the pulse-duration dimension (leaves a time-domain Ramsey signal)
    rabi_inner_product_cos = inner_product_sinusoid(
        np.cos, rabi_freq_hz, ips.mw_pulse_durations_s, rabi_window * np.transpose(dq_signal), axis=1
    )

    if ips.subtract_mean:
        rabi_inner_product_cos = rabi_inner_product_cos - np.mean(rabi_inner_product_cos)

    # Perform the inner product along the evolution-time dimension
    ramsey_window = windows.get_window(ips.ramsey_window, len(ips.free_evolution_times_s))
    double_inner_product_cos_cos = inner_product_sinusoid(
        np.cos, ramsey_freq_hz, ips.free_evolution_times_s, ramsey_window * rabi_inner_product_cos, axis=0
    )

    return double_inner_product_cos_cos


# Calculate I(rabi_freq, ramsey_freq) for a range of values of ramsey_freq
def double_cosine_inner_product_vs_ramsey(
    dq_ms0_fluorescence: NDArray,
    rabi_freq_hz: float,
    ramsey_freq_range_hz: NDArray,
    inner_product_settings: InnerProductSettings,
):
    inner_product_vs_ramsey = []
    for ramsey_freq_hz in ramsey_freq_range_hz:
        inner_product_vs_ramsey.append(
            double_cosine_inner_product(dq_ms0_fluorescence, rabi_freq_hz, ramsey_freq_hz, inner_product_settings)
        )
    return np.array(inner_product_vs_ramsey)

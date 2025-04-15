from lmfit import Model, Parameters, report_fit
from lmfit.model import ModelResult
from lmfit.models import LorentzianModel
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from bff_simulator.constants import gammab, f_h

HZ_TO_MHZ = 1e-6
T_TO_NT = 1e9


def offset(x: float, offset_value: float) -> float:
    return offset_value


def lorentzian(x: float, center: float, fwhm: float, amplitude: float) -> float:
    return amplitude / (1 + 4 * (x - center) ** 2 / fwhm**2)


def set_up_three_peak_model(
    zero_peak_amplitude: float,
    signal_peak_amplitude: float,
    rightmost_peak_location: float,
    t2star_s: float,
    constrain_same_width: bool = True,
    allow_zero_peak: bool = True,
) -> tuple:
    peak0 = Model(lorentzian, prefix="p0_")
    peak1 = Model(lorentzian, prefix="p1_")
    peak2 = Model(lorentzian, prefix="p2_")
    zero_peak = Model(lorentzian, prefix="zero_")
    offset_model = Model(offset)
    three_peak_model = peak0 + peak1 + peak2 + offset_model + zero_peak
    params = three_peak_model.make_params()
    params["offset_value"].value = 0
    for i in range(3):
        params[f"p{i}_amplitude"].value = signal_peak_amplitude
        if i == 0:
            params[f"p{i}_center"].value = rightmost_peak_location
            params[f"p{i}_fwhm"].value = 2 / (np.pi * t2star_s)
        else:
            params[f"p{i}_center"].expr = f"abs(p0_center - 2*{i}*{f_h})"
            if constrain_same_width:
                params[f"p{i}_fwhm"].expr = f"p0_fwhm"
            else:
                params[f"p{i}_fwhm"].value = 2 / (np.pi * t2star_s)

    params[f"zero_fwhm"].value = 2 / (np.pi * t2star_s)
    if allow_zero_peak:
        params["zero_amplitude"].value = zero_peak_amplitude
        params["zero_amplitude"].vary = True
    else:
        params["zero_amplitude"].value = 0
        params["zero_amplitude"].vary = False
    params["zero_center"].value = 0
    params["zero_center"].vary = False
    return three_peak_model, params


def fit_constrained_hyperfine_peaks(
    ramsey_freqs_range_hz: NDArray,
    double_inner_product_cos_cos: NDArray,
    t2star_s: float = 2e-6,
    height_factor: float = 1 / 3,
    prominence_factor: float = 1 / 3,
    constrain_same_width: bool = True,
    allow_zero_peak: bool = True,
) -> ModelResult:
    max_inner_product = max(double_inner_product_cos_cos)
    min_inner_product = min(double_inner_product_cos_cos)

    peaks = find_peaks(
        -double_inner_product_cos_cos,
        height=(-min_inner_product) * height_factor,
        prominence=(-min_inner_product) * prominence_factor,
    )
    rightmost_peak_location = ramsey_freqs_range_hz[peaks[0][-1]]

    three_peak_model, params = set_up_three_peak_model(
        max_inner_product, min_inner_product, rightmost_peak_location, t2star_s, constrain_same_width, allow_zero_peak
    )

    result = three_peak_model.fit(double_inner_product_cos_cos, params, x=ramsey_freqs_range_hz)
    return result


def plot_fit_vs_inner_product(
    ramsey_freqs_range_hz: NDArray, double_inner_product_cos_cos: NDArray, fit_result: ModelResult
) -> None:
    fit_peaks = extract_fit_centers(fit_result)
    plt.vlines(
        HZ_TO_MHZ * fit_peaks,
        [max(double_inner_product_cos_cos)],
        [min(double_inner_product_cos_cos)],
        label="fitted peaks",
        color="red",
    )
    plt.plot(HZ_TO_MHZ * ramsey_freqs_range_hz, double_inner_product_cos_cos, label="cos-cos inner product")
    plt.plot(HZ_TO_MHZ * ramsey_freqs_range_hz, fit_result.best_fit, label="hyperfine-constrained fit")
    plt.xlabel("Inner product Ramsey frequency (MHz)")
    plt.ylabel("Double cosine inner product (a.u.)")
    plt.legend()


def extract_fit_centers(fit_result: ModelResult) -> NDArray:
    fit_peaks = []
    for i in range(3):
        fit_peaks.append(fit_result.best_values[f"p{i}_center"])
    return np.array(fit_peaks)


def fit_vs_eigenvalue_error_nT(fit_result: ModelResult, larmor_freqs_hz: NDArray):
    fit_peaks = extract_fit_centers(fit_result)
    eigenvalue_error_nT = T_TO_NT * (np.sort(np.array(fit_peaks)) - np.sort(larmor_freqs_hz)) / (2 * gammab)
    return eigenvalue_error_nT

from lmfit import Model, Parameters, report_fit
from lmfit.model import ModelResult
from lmfit.models import LorentzianModel
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from bff_simulator.constants import gammab, f_h
from bff_simulator.homogeneous_ensemble import NVOrientation

HZ_TO_MHZ = 1e-6
T_TO_NT = 1e9
N_HYPERFINE = 3

def offset(x: float, offset_value: float) -> float:
    return offset_value

def lorentzian(x: float, freq: float, fwhm: float, amplitude: float) -> float:
    return amplitude / (1 + 4 * (x - freq) ** 2 / fwhm**2)

def decaying_cosine(x: float, freq: float, decay_time: float, amplitude: float, phase: float) -> float:
    return amplitude * np.exp(-x/decay_time)*np.cos(2*np.pi*freq*x + phase)

def set_up_three_cos_model(time_domain_ramsey_signal:NDArray, freq_guesses:NDArray, t2star_s:float, fix_phase_to_zero=False, constrain_same_decay=False, constrain_hyperfine_freqs=False):
    signal_amplitude = (max(time_domain_ramsey_signal) - min(time_domain_ramsey_signal))/2
    model = Model(offset)
    for i in range(N_HYPERFINE):
        model += Model(decaying_cosine, prefix=f"p{i}_")
    params = model.make_params()
    params["offset_value"].value = np.mean(time_domain_ramsey_signal)
    for i in range(N_HYPERFINE):
        params[f"p{i}_amplitude"].value =  -signal_amplitude/N_HYPERFINE
        params[f"p{i}_phase"].value = 0
        params[f"p{i}_decay_time"].value = t2star_s/2
        params[f"p{i}_freq"].value = freq_guesses[i]
        if fix_phase_to_zero:
            params[f"p{i}_phase"].vary = False
        if i>0:
            if constrain_hyperfine_freqs:
                params[f"p{i}_freq"].expr = f"abs(p0_freq - 2*{i}*{f_h})"
            if constrain_same_decay:
                params[f"p{i}_decay_time"].expr = "p0_decay_time"
    return model, params

def fit_three_cos_model(evolution_times_s:NDArray, time_domain_ramsey_signal: NDArray, freq_guesses:NDArray, t2star_s:float, fix_phase_to_zero:bool=False, constrain_same_decay:bool=False, constrain_hyperfine_freqs:bool=False):
    model, params = set_up_three_cos_model(time_domain_ramsey_signal, freq_guesses, t2star_s, fix_phase_to_zero=fix_phase_to_zero, constrain_same_decay=constrain_same_decay, constrain_hyperfine_freqs=constrain_hyperfine_freqs)
    time_domain_result = model.fit(time_domain_ramsey_signal, params, x=evolution_times_s)
    return time_domain_result

def set_up_three_peak_model(
    zero_peak_amplitude: float,
    signal_peak_amplitude: float,
    rightmost_peak_location: float,
    t2star_s: float,
    constrain_same_width: bool = True,
    allow_zero_peak: bool = True,
) -> tuple:
    
    three_peak_model = Model(offset) + Model(lorentzian, prefix="zero_")
    for i in range(N_HYPERFINE):
        three_peak_model+= Model(lorentzian, prefix=f"p{i}_")

    params = three_peak_model.make_params()
    params["offset_value"].value = 0
    for i in range(N_HYPERFINE):
        params[f"p{i}_amplitude"].value = signal_peak_amplitude
        if i == 0:
            params[f"p{i}_freq"].value = rightmost_peak_location
            params[f"p{i}_fwhm"].value = 2 / (np.pi * t2star_s)
        else:
            params[f"p{i}_freq"].expr = f"abs(p0_freq - 2*{i}*{f_h})"
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
    params["zero_freq"].value = 0
    params["zero_freq"].vary = False
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
    return np.array([fit_result.best_values[f"p{i}_freq"] for i in range(N_HYPERFINE)])

def extract_fit_centers_all_orientations(fit_results: list[type[ModelResult]]) -> NDArray:
    return np.array([extract_fit_centers(fit_results[orientation]) for orientation in NVOrientation])

def extract_fit_center_stderrs_nT(fit_result: ModelResult) -> NDArray:
    return T_TO_NT*np.array([fit_result.params[f"p{i}_freq"].stderr for i in range(N_HYPERFINE)])/(2*gammab)

def extract_fit_center_stderrs_all_orientations_nT(fit_results: list[type[ModelResult]]) -> NDArray:
    return np.array([extract_fit_center_stderrs_nT(fit_results[orientation]) for orientation in NVOrientation])

def fit_vs_eigenvalue_error_nT(fit_result: ModelResult, larmor_freqs_hz: NDArray):
    fit_peaks = extract_fit_centers(fit_result)
    eigenvalue_error_nT = []
    for peak in fit_peaks:
        larmor_freq_index = np.abs(peak-larmor_freqs_hz).argmin()
        eigenvalue_error_nT.append(T_TO_NT*(peak - larmor_freqs_hz[larmor_freq_index])/(2*gammab))
    #eigenvalue_error_nT_old = T_TO_NT * (np.sort(np.array(fit_peaks)) - np.sort(larmor_freqs_hz)) / (2 * gammab)
    return np.array(eigenvalue_error_nT)

def fit_vs_eigenvalue_error_all_orientations_nT(fit_results: list[type[ModelResult]], larmor_freqs_all_orientations_hz: NDArray) -> NDArray:
    return np.array([fit_vs_eigenvalue_error_nT(fit_results[nv], larmor_freqs_all_orientations_hz[nv]) for nv in NVOrientation])

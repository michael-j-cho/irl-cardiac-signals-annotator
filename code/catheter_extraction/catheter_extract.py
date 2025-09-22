from catheter_configs import ao_ranges, ac_ranges, ao_diff3_flag, ac_diff3_flag

import numpy as np
import os
from pathlib import Path
import scipy.io
from scipy.signal import butter, filtfilt, find_peaks

# Initialize directories
RAW_DIR = Path("./data/raw")
RAW_FILES = [(RAW_DIR / f) for f in os.listdir(RAW_DIR) if f.endswith(".mat")]
RAW_FILES.sort()

FIG_DIR = Path("./figures")
PROCESSED_DIR = Path("./data/processed")

FS = 2000


def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a lowpass Butterworth filter to the data.

    Parameters:
    - data: Input signal (1D or 2D array)
    - cutoff: Cutoff frequency (Hz)
    - fs: Sampling frequency (Hz)
    - order: Order of the filter

    Returns:
    - Filtered signal
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=-1)


def ao_cath_peaks(cath_diff, ao_begin, ao_end):
    """
    Extract aortic catheter peaks from the given signal.

    Parameters:
    - cath_diff: Differentiated catheter signal
    - ao_begin: Start index for aortic range
    - ao_end: End index for aortic range

    Returns:a
    - List of indices corresponding to aortic catheter peaks
    """
    ao_cath = []
    for beat in cath_diff:
        peaks, heights = find_peaks(beat[ao_begin:ao_end], height=0)
        if len(heights["peak_heights"]) < 1:
            ao_cath.append(0)
            continue
        max_peak_index = peaks[np.argmax(heights["peak_heights"])]
        ao_cath.append(max_peak_index + ao_begin)
    return ao_cath


def ac_cath_peaks(cath_diff, ac_begin, ac_end, select_peak=-2):
    """
    Extract atrial catheter peaks from the given signal.

    Parameters:
    - cath_diff: Differentiated catheter signal
    - ac_begin: Start index for atrial range
    - ac_end: End index for atrial range
    - select_peak: Index of the peak to select (default is second highest)

    Returns:
    - List of indices corresponding to atrial catheter peaks
    """
    ac_cath = []
    for beat in cath_diff:
        peaks, heights = find_peaks(-beat[ac_begin:ac_end], height=0)

        if len(heights["peak_heights"]) > 1:
            second_highest_peak_index = peaks[np.argsort(heights["peak_heights"])[select_peak]]
            ac_cath.append(second_highest_peak_index + ac_begin)
        else:
            ac_cath.append(0)  # Append 0 if there is no second highest peak

    return ac_cath


def save_file(ao_cath, ac_cath, ap_beats, scg_beats, ecg_beats, file_name):
    """
    Save extracted catheter timing data to a .mat file.

    Parameters:
    - ao_cath: Aortic catheter peaks
    - ac_cath: Atrial catheter peaks
    - ap_beats: Catheter signal beats
    - scg_beats: SCG signal beats
    - ecg_beats: ECG signal beats
    - file_name: Name of the file to save the data
    """
    save_dict = {
        "ac_cath": ac_cath,
        "ao_cath": ao_cath,
        "ap_beats": ap_beats,
        "scg_beats": scg_beats,
        "ecg_beats": ecg_beats,
    }
    scipy.io.savemat(PROCESSED_DIR / file_name, save_dict)


def extract_catheter_timings():
    """
    Process raw catheter data files to extract aortic and atrial catheter timings.

    Iterates through all raw data files, applies filtering and peak detection,
    and saves the extracted timings to processed files.
    """
    for j in range(len(RAW_FILES)):
        # Load data from file
        data = scipy.io.loadmat(RAW_FILES[j])

        # ecg = data["ecgBeats"]
        scg = data["scg_beats_interval"]
        cath = data["ap_beats_interval"]
        sqi = data["sqi_values_interval"][0]
        
        cath = lowpass_filter(cath, 25, FS)
        
        cath_diff = lowpass_filter(np.diff(cath), 25, FS)
        cath_diff = np.diff(cath)
        cath_diff2 = np.diff(cath_diff)
        cath_diff3 = np.diff(cath_diff2)

        current_file = float(str(os.path.basename(RAW_FILES[j])[1:4]))
        ao_begin = ao_ranges[current_file][0]
        ao_end = ao_ranges[current_file][1]
        ac_begin = ac_ranges[current_file][0]
        ac_end = ac_ranges[current_file][1]

        if ao_diff3_flag[current_file] == False:
            ao_cath = ao_cath_peaks(cath_diff2, ao_begin, ao_end)
        else:
            ao_cath = ao_cath_peaks(cath_diff3, ao_begin, ao_end)

        if ac_diff3_flag[current_file]:
            ac_cath = ac_cath_peaks(cath_diff3, ac_begin, ac_end)
        else:
            ac_cath = ac_cath_peaks(cath_diff2, ac_begin, ac_end, -1)
        
        save_name = str(os.path.basename(RAW_FILES[j]))[:5] + "final.mat"
        save_file(ao_cath, ac_cath, cath, scg, ecg, save_name)


def main():
    """
    Main function to initiate catheter timing extraction.
    """
    extract_catheter_timings()


if __name__ == "__main__":
    main()

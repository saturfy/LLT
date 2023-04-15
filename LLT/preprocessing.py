"""This sub package collects helper functions and classes for preprocessing ECG data samples. It handles loading, cleaning, resampling and QRS complex detection. 
The package also contains useful functions for evaluating ECG classificaiton tasks. """

# IMPORTS

import scipy.io
import scipy.signal as sg
import numpy as np


def Load_VPNet_data(file):
    """
    Loads the VPNet dataset from matlab format and returns the
    samples and the labels.
    labels:
        0 : ectopic
        1 : normal

    Parameters
    ----------
    file : string
        name of the .mat file to be loaded

    Returns
    -------
    (np.ndarray, np.ndarray)
        first is the numpy array of samples and the second is
        the numpy array of the corresponding labels

    """
    data_dict = scipy.io.loadmat(file)

    return data_dict["samples"], data_dict["labels"].T[0]


def Resample(samples, read_in_freq, sampling_freq):
    """
    Resamples the data samples to have a given sampling frequency.
    
    Parameters
    ----------
    samples : list
        list of data samples which are numpy.ndarrays
    
    read_in_freq : float
        The sampling frequency of the data samples
    
    sampling_freq : float
        The sampling frequency of the resampled data
        
    Returns
    -------
    numpy.ndarray
        numpy array of the transformed data samples
    """
    resampled_data = []
    for beat in samples:
        resampled_data.append(sg.resample(beat, int(len(beat) * sampling_freq/read_in_freq)))

    return np.array(resampled_data)


def denoise(sample, sampling_freq=250, cutoff_high=0.5, cutoff_low=20, powerline=50, order=3):
    """
    Applies a low-pass, a high-pass and a notch filter to filter out the noise from the ECG signal.

    Parameters
    ----------
    data : numpy.ndarray
        ECG signal array. shape=(sample length,)
    sampling_freq : float
        The sampling frequency of the data
    cutoff_high : float
        The cutoff frequency for the high-pass filter in Hz. (frequencies below this are supressed)
    cutoff_low : float
        The cutoff frequency for the low-pass filter in Hz. (frequencies above this are supressed)
    powerline : float
        The frequency for the notch filter. (frequencies around this are supressed)
    order : int
        The order of the filter. The applied filter order is twice as large because the filters are applied
        twice (left and than right)


    Returns
    -------
    numpy.ndarray
        The filtered signal, an array.

    """
    # normalize frequencies
    nyq = 0.5 * sampling_freq
    norm_cutoff_high = cutoff_high / nyq
    norm_cutoff_low = cutoff_low / nyq
    norm_powerline = powerline / nyq

    # high pass filter
    b_h, a_h = sg.butter(order, norm_cutoff_high, btype='high', analog=False, output='ba')

    # low pass filter
    b_l, a_l = sg.butter(order, norm_cutoff_low, btype='low', analog=False, output='ba')

    # notch filter
    q = 30  # quality
    b_n, a_n = sg.iirnotch(norm_powerline, q)

    # --- Apply filters ---

    # high pass
    x = sg.filtfilt(b_h, a_h, sample)

    # low pass
    y = sg.filtfilt(b_l, a_l, x)

    # notch
    z = sg.filtfilt(b_n, a_n, y)

    return z


class Peak_data:
    """
    Collects the peak data in a deatailed way to esily generate datasets.

    Attributes
    ----------
        zpk : list
            every entry is a list. First element is the list of arrays containing the peaks
            second element is the index of the sample (in the original dataset)
            where they are coming from
            This only collects data where no peaks were found. ( all first entry is [])

        lpk : list
            every entry is a list. First element is the list of arrays containing the peaks
            second element is the index of the sample (in the original dataset)
            where they are coming from
            This only collects data where the peaks come from samples which contained only
            one peak per sample. (lone peaks)

        mpks : list
            every entry is a list. First element is the list of arrays containing the peaks
            second element is the index of the sample (in the original dataset)
            where they are coming from
            This only collects data where the peaks come from samples which contained
            multiple peaks per sample.

        lpk_std : list
            same as lpk just only contains the peeks which has the same size as the window

        mpks_std : list
            same as mpks just only contains the peeks which has the same (standard) size
            as the window

    Note
    ----
    Detailed structure of the zpk, lpk, mpks etc.. data:
    mpks = [[[array1, array2, array3 ...], index], [[array1, array2, array3 ...], index], ... ]
    where array# holds the QRS peak data for every peak found in the sample at database[index].
    To reach the second detected peak data in the 10th element: mpks[10][0][1]
    firs index: specifies sample
    second index: specifies peaks or original index
    third index: specifies peak

    """
    def __init__(self, dataset, peak_height=0.5, normalize="max", window=[10, 10]):
        """
        Parameters
        ----------
        dataset : numpy.ndarray
            Array of the input signals. shape = (number of samples in the set, length of the samples)
        peak_height : float
            maxima larger than this value are detected as peaks
        normalize : string
            sets how the data is normalized before searching for peaks
            none: raw signal is used
            max: mean is substracted than normed with the max of the signal used for peak finding
        window : list
            list of integers: [int1, int2]. Part of the signal is cut out around the middle taking int1/int2
            number of points from the left/right of middle point.
            if None nothing will be cut out, it leaves the original signal.

        Returns
        -------
        Peak_data class instance.
        """
        window_size = 1 + window[0] + window[1]
        self.zpk = []
        self.lpk = []
        self.lpk_std = []
        self.mpks = []
        self.mpks_std = []

        # --- Find the peaks in the dataset ---
        peak_list = []

        for sample in dataset:
            sub_peaks = self._export_QRS_peaks(sample, peak_height=peak_height, normalize=normalize, window=window)
            peak_list.append(sub_peaks)

        # --- load data and separate into datasets ---
        for indx, sub_beats in enumerate(peak_list):
            if len(sub_beats) == 1:
                self.lpk.append([[sub_beats[0]], indx])
            elif len(sub_beats) > 1:
                self.mpks.append([sub_beats, indx])
            else:
                self.zpk.append([sub_beats, indx])

        # --- separate further based on length ---
        # one peaks
        for peak_dat in self.lpk:
            if len(peak_dat[0][0]) == window_size:
                self.lpk_std.append(peak_dat)

        # multiple peaks
        for peak_data in self.mpks:
            # empty list for the correct data
            peak_data_std = [[], peak_data[1]]

            # add peaks if they have the correct length
            for one_peak in peak_data[0]:
                if len(one_peak) == window_size:
                    peak_data_std[0].append(one_peak)

            # if nothing were added we skip
            if peak_data_std[0] is not []:
                self.mpks_std.append(peak_data_std)

    def _export_QRS_peaks(self, sample, peak_height=0.5, normalize="max", window=[10, 10]):
        """
        Finds QRS peaks in the sample.

        Parameters
        ----------
        sample : numpy.ndarray
            The input signal shape = (signal lenght, )
        peak_height : float
            maxima larger than this value are detected as peaks
        normalize : string
            sets how the data is normalized before searching for peaks
            none: raw signal is used
            max: mean is substracted than normed with the max of the signal used for peak finding
            std: standardize the sample
        window : list
            list of integers: [int1, int2]. Part of the signal is cut out around the middle taking int1/int2
            number of points from the left/right of middle point.
            if None nothing will be cut out, it leaves the original signal.

        Returns
        -------
        list
            list of [y, ... y] lists where y is numpy.ndarray. ys are the cut out signal values
            around the peak, according to the choosen window. The resulting list contains at every entry
            the list of peaks found in the in a given sample contained the input dataset.
        """

        # --- exporting peaks ---

        # normalize
        if normalize == "max":
            beat = (sample - sample.mean())
            beat = beat / max(abs(beat))
        elif normalize == "std":
            beat = (sample - sample.mean())
            beat = beat / sample.std()
        else:
            beat = sample

        # find peak locations in one sample
        beat_indxs, _ = sg.find_peaks(beat, height=peak_height)
        beat_indxs_neg, _ = sg.find_peaks(-beat, height=peak_height)
        beat_indxs = np.concatenate((beat_indxs, beat_indxs_neg))

        # All the peaks found will be stored in a list
        sub_peaks = []

        for sub_beat in beat_indxs:
            # cutting the window from the signal around the peak
            # if the window is too big, we cut until the signal ends

            # the indexes of the interval which is selected from the sample by the window
            min_ind = sub_beat-window[0]
            max_ind = sub_beat+1+window[1]

            sample_min_ind = 0
            sample_max_ind = len(beat) - 1

            # if everything is fine and this interval can fit into the sample size
            if (min_ind >= sample_min_ind) and (max_ind <= (sample_max_ind + 1)):

                # if the last element of the sample is included in the window the indexing would be outside of range
                if max_ind == (sample_max_ind + 1):
                    y = beat[min_ind:]
                else:
                    y = beat[min_ind: max_ind]

            # if the left side of the window is outside of the sample and the right side is inside
            elif (min_ind < sample_min_ind) and (max_ind <= (sample_max_ind + 1)):
                # cut the window size by selecting the part which fits the sample
                min_ind = 0

                # if the last element of the sample is included in the window the indexing would be outside of range
                if max_ind == (sample_max_ind + 1):
                    y = beat[min_ind:]
                else:
                    y = beat[min_ind: max_ind]

            # if the left side of the window is inside of the sample and the right side is outside
            elif (min_ind >= sample_min_ind) and (max_ind > (sample_max_ind + 1)):
                # in this case we always have to keep all the elements right from the beat peak
                y = beat[min_ind:]

            # if both sides of the window are outside of the sample
            elif (min_ind < sample_min_ind) and (max_ind > (sample_max_ind + 1)):
                # we have to keep all elements of the sample
                y = beat[:]

            # save data
            sub_peaks.append(y)

        return sub_peaks

    def generate_peak_dataset(self, dataset_labels):
        """
        Generates a data set, label set pair from the standard length peaks found in the originial dataset.
        Samples from the original dataset where multiple  or zer peaks were found are left out.

        Parameters
        ----------
        dataset_labels : numpy.ndarray
            The lables corresponding to the dataset used to initialize the class.
            Array containing 0 or 1, where
            0: ectopic
            1: normal.
            shape=(number of samples, )

        Returns
        -------
        (numpyn.ndarray, numpy.ndarray)
        First array is the array of peak data
        second array containts the labels:
        (sample array, label array)

        """
        samples = [peak_data[0][0] for peak_data in self.lpk_std]
        labels = [dataset_labels[peak_data[1]] for peak_data in self.lpk_std]

        return np.array(samples), np.array(labels)


def general_predict(samples, linear_model, classifier,
                    feature_normalize=0.95,
                    peak_height=0.5,
                    normalize="max",
                    window=[10, 10]):
    """
    Combines the peak search and peak classifier algorithm into one single
    predictor. The peak classifier can only work on prepared peaks, however
    this function works on full samples and handles the peak search inside.
    If peak search returns something different than one peak per sample, the
    method automatically classifies it as ectopic.

    Parameters
    ----------
    samples : numpy.ndarray
        shape=(number of samples, lenght of samples)
        The collection of time series samples which may or
        may not contain  a peak.
    linear_model : linear_law class instance
        The linear model of the dataset which is used to
        generate the features.
    classifier : object
        A classifier which is trained on linear law generated
        features of the peaks.
        It must have a predict method which accepts an array
        of featrues shape=(samples, number of features)
    feature_normalize : float
        The normalization parameter for the feature generation.
    peak_height : float
        maxima larger than this value are detected as peaks
    normalize : string
        sets how the data is normalized before searching for peaks
        none: raw signal is used
        max: mean is substracted than normed with the max of the signal used for peak finding
    window : list
        list of integers: [int1, int2]. Part of the signal is cut out around the middle taking int1/int2
        number of points from the left/right of middle point.
        if None nothing will be cut out, it leaves the original signal.

    Returns
    -------
    numpy.ndarray
    Predicted classes of the input samples
    0: ectopic
    1: normal
    shape=(number of samples,)

    Note
    ----
    This function only combines the peak detection and peak classification
    algorithms into one. It does not guarantee consistency, the user must make
    sure that:
    1) The peak search settings are identical to what was used during the
    sample preparation to train the classifier
    2) The classifier is trained on the features generated by the same linear model
    which was given as an argument
    3) feature normalization is the same as it was used for the classifier training.

    """
    # --- peak search ---
    samples_peak_data = Peak_data(samples, peak_height=peak_height, normalize=normalize, window=window)
    # selecting lone peaks
    samples_peaks = np.array([sub_peak[0][0] for sub_peak in samples_peak_data.lpk_std])
    idx_list = [sub_peak[1] for sub_peak in samples_peak_data.lpk_std]

    # --- feature transform ---
    tf_samples_peaks = linear_model.feature_transform(samples_peaks, normalize=feature_normalize)

    # --- predict peaks ---
    predictions = classifier.predict(tf_samples_peaks)

    # --- organize the result according to the original input ---
    full_predictions = np.zeros(len(samples))

    full_predictions[idx_list] = predictions

    return full_predictions


def performance_metrics(CM, text=True):
    """
    Generates binary classification performance metrics from the confusion matrix.
    The following label convention is used for convenience
    0: ectopic class
    1: normal class

    Parameters
    ----------
    CM : numpy.ndarray
        Cunfusion matrix, shape=(2,2)
    text : boolean
        If true prints the peformance metrics on the console.

    Returns
    -------
    dict
    The performance metrics in a dict

    """
    # total accuracy
    Ta = (CM[0][0] + CM[1][1])/CM.sum()
    # Normal Sensitivity
    NSe = (CM[0][0])/CM[0].sum()
    # Ectopic Sensitivity
    ESe = (CM[1][1])/CM[1].sum()
    # normal positive predictibility
    Npp = (CM[0][0])/(CM[0][0]+CM[1][0])
    # ectopic positive predictibility
    Epp = (CM[1][1])/(CM[1][1]+CM[0][1])

    if text:
        print(f"                 Total accuracy :  { Ta :.4f}")
        print("- - - - - - - - - - - - - - - - - - - - - ")
        print(f"             Normal sensitivity :  { NSe :.4f}")
        print(f"            Ectopic sensitivity :  { ESe :.4f}")
        print(f" Normal positive predictibility :  { Npp :.4f}")
        print(f"Ectopic positive predictibility :  { Epp :.4f}")

    resdict = {
        "Total_accuracy": Ta,
        "Normal_sensitivity": NSe,
        "Ectopic_sensitivity ": ESe,
        "Normal_positive_predictibility": Npp,
        "Ectopic_positive_predictibility": Epp
    }

    return resdict

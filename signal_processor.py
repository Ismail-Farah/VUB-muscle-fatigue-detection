import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy import fftpack
from biosignalsnotebooks import detect as bsnb
import plotting as p
import filtering_config as fc


# Constants
ENGINE = 'python'
TXT_SUFFIX = '.txt'
CSV_SUFFIX = '.csv'
COLUMNS = [
    'emg',
    'time',
]


def add_time(data_raw):
    data_raw['time'] = np.arange(len(data_raw)) + 1


def split_to_channels(data):
    data[['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8',
          'emg9', 'emg10', 'emg11', 'emg12', 'emg13', 'emg14', 'emg15', 'emg16']] = \
        data['emg'].str.split(';', 15, expand=True)
    # Remove the ';' from channel 16     
    data['emg16'] = data['emg16'].str.replace(";", "")


def import_csv_data(data_file_name=None, separator=None):
    """ This function reads all the csv datasets.
    Input: dataframe that also has a column 'time'
    separator: what separates the data (e.g. "," or "\t")
    Output: continuous time over all datasets
   """
    # Creating an empty Dataframe with column names only
    data_raw = pd.DataFrame(columns=COLUMNS)

    # Create string of the path
    data_name_string = data_file_name + CSV_SUFFIX
    data_raw = read_data(data_name_string, data_raw, separator)
    add_time(data_raw)
    split_to_channels(data_raw)
    data_raw = convert_to_float(data_raw, channels=16)

    return data_raw


def import_data(data_file_name, separator, size=1):
    """ This function is when you put together several datasets,
    but each dataset always starts with a time of 0.
    Input: dataframe that also has a column 'time'
    separator: what separates the data (e.g. "," or "\t")
    size: int, in case we have multiple files (e.g. "example1.txt" & "example2.txt")
    Output: continuous time over all datasets
   """
    # Creating an empty Dataframe with column names only
    data_raw = pd.DataFrame(columns=COLUMNS)

    if size > 1:
        # Read all data files
        for i in range(size):
            # create string of the path (e.g. "example1.txt" & "example2.txt")
            data_name_string = create_file_name(data_file_name, i)
            data_raw = read_data(data_name_string, data_raw, separator)
    else:
        # Create string of the path
        data_name_string = data_file_name + TXT_SUFFIX
        data_raw = read_data(data_name_string, data_raw, separator)

    # Timing might need to change as the appended data start from 0 again
    data_output = normalize_time(data_raw)
    return data_output


def create_file_name(data_file_name, i):
    return data_file_name + str(i + 1) + TXT_SUFFIX


def read_data(data_name_string, data_raw, separator):
    """import data and put data in one variable"""
    return pd.concat([data_raw, pd.read_csv(
        data_name_string,
        sep=separator,
        engine=ENGINE, names=COLUMNS, skiprows=6,
        skipfooter=0
    )])


def normalize_time(data):
    a = list(data.iloc[:]['time'])
    b = list(data.iloc[:]['time'])

    for j in range(len(a) - 1):
        if a[j] > a[j + 1]:
            if b[j] > b[j + 1]:
                offset = a[j] - a[j + 1] + 1
                a[j + 1] = offset + a[j + 1]
                j += 1
            else:
                a[j + 1] = offset + a[j + 1]
                j += 1

    emg_data = convert_to_float(data)

    updated_data = pd.DataFrame({'emg': emg_data, 'time': a})
    updated_data.reset_index(inplace=True, drop=True)

    return updated_data


def convert_to_float(data, channels=None):
    """
    Convert the emg data readings to float value
    """
    if channels is None:
        emg_data = to_float(data).emg
    else:
        # In case of multiple channels in one file
        for i in range(channels):
            column = "emg" + str(i + 1)
            emg_data = to_float(data, column=column)

    return emg_data


def to_float(data, column='emg'):
    emg_data = list(data.iloc[:][column])
    for index in range(len(emg_data)):
        value = str(emg_data[index]).replace(',', '.')
        emg_data[index] = float(value)
    data[column] = emg_data
    return data


def remove_mean(raw_emg, time, filter_config):
    if filter_config.fix_baseline_shift:
        emg_corrected_mean = raw_emg - np.mean(raw_emg)
        p.plot_comparison(time=time,
                          emg_before=raw_emg, plot_title_before='Mean offset present',
                          emg_after=emg_corrected_mean, plot_title_after='Mean-corrected values',
                          )
        return emg_corrected_mean
    else:
        return raw_emg


def filter_signal(emg_raw_data, time, filter_config):
    a, b = fc.create_bandpass_filter(filter_config=filter_config)
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b, a, emg_raw_data)
    p.plot_comparison(time=time,
                      emg_before=emg_raw_data, plot_title_before='Unfiltered EMG',
                      emg_after=emg_filtered, plot_title_after='Filtered EMG',
                      )

    return emg_filtered


def rectify(emg_filtered, time):
    # Process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    # plot comparison of un-rectified vs rectified EMG
    p.plot_comparison(time=time,
                      emg_before=emg_filtered, plot_title_before='Un-rectified EMG',
                      emg_after=emg_rectified, plot_title_after='Rectified EMG',
                      )

    return emg_rectified


def clean_up_emg_signal(time, emg_data, filter_configs, show=False):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut-off frequency
    low: low-pass cut-off frequency.
    sampling_frequency: sampling frequency of the emg data
    """
    # Normalise cut-off frequencies to sampling frequency
    a1, b1 = fc.create_bandpass_filter(filter_configs)

    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg_data)

    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)

    a2, b2, low_pass = fc.create_low_pass_filter(filter_config=filter_configs)
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

    p.plot_signal_transformations(emg_data, emg_rectified, emg_envelope, time, filter_configs, show)

    return emg_filtered, emg_envelope


def get_pluses(signal):
    threshold = 1.2 * np.mean(signal) + 0.8 * np.std(signal, ddof=1)

    # Find segments
    length = len(signal)
    start = np.nonzero(signal > threshold)[0]
    stop = np.nonzero(signal < threshold)[0]

    segments = np.union1d(np.intersect1d(start - 1, stop),
                          np.intersect1d(start + 1, stop))

    if np.any(segments):
        if segments[-1] >= length:
            segments[-1] = length - 1

    return segments


def convert_segments_to_milliseconds(segments_starts, segments_ends):
    segments_starts = [int(x * 1000) for x in segments_starts]
    segments_ends = [int(x * 1000) for x in segments_ends]
    return segments_starts, segments_ends


def detect_segments(signal, filtering_configs, display=False):
    """
    Using Teager Kaiser Energy Operator
    :param signal: emg signal
    :param filtering_configs: an Object that holds all the filtering configurations
    :param display: show the plot
    :return: starts, ends & processed signal
    """
    segments_starts, segments_ends, smooth_signal, threshold_level = bsnb. \
        detect_emg_activations(signal,
                               filtering_configs.sampling_frequency,
                               smooth_level=filtering_configs.window_size,
                               threshold_level=filtering_configs.threshold_level,
                               time_units=True,
                               volts=False,
                               resolution=None,
                               device='CH0',
                               plot_result=display)

    segments_starts, segments_ends = convert_segments_to_milliseconds(segments_starts, segments_ends)

    return segments_starts, segments_ends, smooth_signal


def get_moving_average(emg_data, time, window_size):
    moving_averages_emg = []
    moving_averages_time = []
    start_index = int(window_size / 2)
    end_index = len(emg_data) - int(window_size / 2)
    time_list = time.tolist()
    for i in range(start_index, end_index, window_size):
        point_average = np.mean(emg_data[i:i + window_size])
        moving_averages_emg.append(point_average)
        moving_averages_time.append(time_list[i])

    return moving_averages_time, moving_averages_emg


def get_signal_segments(emg_data, time, filtering_configs, show=False):
    """
    This function is detecting the activities of the muscle
    Arguments:
    emg_data: EMG data to process.
    time: The time corresponding to each EMG activity.
    emg_filtered: EMG data filtered.
    window_size: The window size of the moving average window
    show: boolean to show or hide the resulting plots
    Returns:
    data_starts: List of activity start time
    data_ends: List of activities end time
    moving_average: the moving average data
   """
    moving_average_time, moving_average_emg = get_moving_average(emg_data, time, filtering_configs.window_size)

    segments_starts, segments_ends, smooth_signal = detect_segments(emg_data, filtering_configs, display=False)

    p.plot_segments_with_moving_average(emg_data, time,
                                        moving_average_emg, moving_average_time,
                                        segments_starts, segments_ends,
                                        filtering_configs,
                                        show)

    return segments_starts, segments_ends, smooth_signal, moving_average_time, moving_average_emg


def get_starts_ends(pulses):
    segments_starts = [x for i, x in enumerate(pulses) if not i % 2]
    segments_ends = [x for i, x in enumerate(pulses) if i % 2]
    segments = [sorted([x, y]) for x, y in zip(segments_starts, segments_ends)]

    starts = [item[0] for item in segments]
    ends = [item[1] for item in segments]
    return starts, ends


""" 
Full experiment. Process the signal to clean it up from noise,
then compare the signal's strength within each of the muscle flexes 
"""


def get_segments(activity_starts, activity_ends):
    # Sort start and end of contraction
    return sorted(activity_starts + activity_ends)


def split_data(emg_signal, time, segments_number, segment_checked_number, filtering_config):
    # In case we want to process a given part of the whole data file
    segment_start, segment_end = get_checked_part_start_end(emg_signal,
                                                            filtering_config,
                                                            segment_checked_number,
                                                            segments_number)

    emg_signal = emg_signal[:] if segments_number == 1 else emg_signal[segment_start:segment_end]
    time = time if segments_number == 1 else time[segment_start:segment_end]

    return emg_signal, time



def create_fatigue_figure_path(file_name):
    return "./figures/" + file_name[1] + "_fatigue"


def get_checked_part_start_end(emg_signal, filter_config, segment_checked_number, segments_number):
    # Split the data in case 'segments_number' & 'segment_checked_number' are provided
    segment_start = int((len(emg_signal) / segments_number) * (segment_checked_number - 1)) + filter_config.pre_ignore
    segment_end = segment_start + int(len(emg_signal) / segments_number) - filter_config.post_ignore
    return segment_start, segment_end


def show_experiment_info(data, segments_number, time):
    """
    When the data is segmented, this method shows which part of the data is being handled currently
    """
    print("Full experiment time = " + str(data.time.max() / 1000) + " seconds")
    if segments_number != 1:
        print("This experiment time frame starts at " + str(time.min() / 1000) + " seconds to " +
              str(time.max() / 1000) + " seconds")


def calculate_muscle_activities(activity_starts, activity_ends, emg_data,
                                time, checked_flexs=None, show=False):
    power_means = []
    start_power_means = []
    mid_power_means = []
    end_power_means = []
    power_mean_time = []
    for i in range(len(activity_starts)):
        # Get the start, mid and end of a flex
        array, array_start, array_mid, array_end = get_flex_parts(activity_starts, activity_ends, i, emg_data)
        # Skip small pulses which are less than 450 ms
        if len(array) >= 450 :   
            power_start, frequencies_start = get_power(array_start.tolist(), 1000)
            power_mid, frequencies_mid = get_power(array_mid.tolist(), 1000)
            power_end, frequencies_end = get_power(array_end.tolist(), 1000)
            
            b2, a2 = sp.signal.butter(4, 0.08, btype="lowpass")
            filtered_power_start = sp.signal.filtfilt(b2, a2, power_start)
            filtered_power_mid = sp.signal.filtfilt(b2, a2, power_mid)
            filtered_power_end = sp.signal.filtfilt(b2, a2, power_end)
            
            power_start_avg = np.median(filtered_power_start)
            power_mid_avg = np.median(filtered_power_mid)
            power_end_svg = np.median(filtered_power_end)
            power_medians_list = [power_start_avg, power_mid_avg, power_end_svg]
            
            # Save the current power mean with its respective time
            power_means.append(np.median(power_medians_list))
            start_power_means.append(power_start_avg)
            mid_power_means.append(power_mid_avg)
            end_power_means.append(power_end_svg)
            power_mean_time.append(activity_ends[i])
            
            if (i+1) in checked_flexs:
                p.plot_activation_detials(activation_number=i+1,
                                      array=array,
                                      array_start=array_start,
                                      array_mid=array_mid,
                                      array_end=array_end,
                                      power_start=power_start, frequencies_start=frequencies_start,
                                      filtered_power_start=filtered_power_start,
                                      power_mid=power_mid, frequencies_mid=frequencies_mid,
                                      filtered_power_mid=filtered_power_mid,
                                      power_end=power_end, frequencies_end=frequencies_end,
                                      filtered_power_end=filtered_power_end,
                                      power_medians_list=power_medians_list,
                                      show=show)

    return power_mean_time, power_means, start_power_means, mid_power_means, end_power_means


def get_flex_parts(activity_starts, activity_ends, checked_flex_number, emg_filtered):
    """
    Get tge parts of a specific flex from the flexes marked by 'activity_starts' & 'activity_ends'
    :param activity_starts: list of points where the flexes start
    :param activity_ends:  List of points where the flexes stop
    :param checked_flex_number: the current checked flex
    :param emg_filtered: emg filtered data
    :return:
        array: the full flex,
        array_start: the first part of the flex,
        array_mid middle part of the flex,
        array_end: last part of the flex
    """
    array = pd.Series(emg_filtered[activity_starts[checked_flex_number]:
                                   activity_ends[checked_flex_number]])
    # To make sure we are in the activation of the muscle
    # 10% of the length     
    # pre_range_shift = int(len(array)*(10/100))
    pre_range_shift = 100

    # 30% of the length    
    # post_range_shift = int(len(array)*(30/100))
    post_range_shift = 300
    
    # 15% of the length
    # checked_range = int(len(array)*(15/100))
    checked_range = 150
    
    array = array[pre_range_shift:-post_range_shift]
    array_start = array[:checked_range]
    array_mid = array[int(len(array) / 2) - int(checked_range / 2):
                      int(len(array) / 2) + int(checked_range / 2)]
    array_end = array[-checked_range:]
    return array, array_start, array_mid, array_end


def get_power(data, sampling_frequency=1000):
    """ This function does an FFT and returns the power spectrum of data
        Input: Data, sampling frequency
        Output: Power, the respective frequencies of the power spectrum
       """
    sig_fft = fftpack.fft(data)

    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)

    # The corresponding frequencies
    sample_freq1 = fftpack.fftfreq(len(data), d=1 / sampling_frequency)
    frequencies = sample_freq1[sample_freq1 > 0]
    power = power[sample_freq1 > 0]

    return power, frequencies


def apply_fast_fourier_transform(emg_data, sampling_frequency=1000, figure_name="Power Spectrum"):
    power, frequency = get_power(emg_data)
    plt.figure()
    plt.plot(frequency, power)
    plt.title(figure_name)
    plt.plot("Power (a. u.)")
    plt.xlabel("Frequency (Hz)")

    
def add_onsets_class(onsets, activity_starts, activity_ends):
    for i in range(len(activity_starts)):
        for j in range(activity_starts[i], activity_ends[i], 1):
            onsets[j] = 1

    return onsets


def add_segments_to_data_file(data, subject, channel, activity_starts, activity_ends):
    onsets = [0 for _ in range(len(data))]
    data["class"] = add_onsets_class(onsets, activity_starts, activity_ends)
    data.to_csv(subject + "_data/dataset/channel_" + str(channel), sep='\t')
    
    
# This function loads the data, filter it and segment the contraction signals, and then calcuate the median muscle fatigue for each contraction of the loaded data using fast fourier transform.
def process_emg(data_file_name, filtering_config=None, segments_number=1, segment_checked_number=1,
                checked_flexes=None,
                show_intermediate=False, show_fatigue=True):
    # Filtering and cleaning up emg data
    data = import_data(data_file_name, "\t")
    # Fix baseline shift if 'filter_config.fix_baseline_shift' is True
    emg_signal = remove_mean(data.emg, data.time, filtering_config)

    # In case we want to process a given part of the whole data file
    emg_signal, time = split_data(emg_signal, data.time, segments_number, segment_checked_number, filtering_config)

    # Create filtered emg data & emg envelope
    emg_filtered, emg_envelope = clean_up_emg_signal(time, emg_signal, filtering_config, show=show_intermediate)

    activity_starts, activity_ends, smooth_signal, moving_average_time, moving_average_emg = get_signal_segments(
        emg_signal, time, filtering_config, show=False)

    segments = get_segments(data_file_name, activity_starts, activity_ends)

    # Name to save the segments figure
    file_name = data_file_name.split("/")
    p.plot_emg_segments(time=time,
                        emg_raw=emg_signal,
#                       TODO: use the filtered signal when ploting the segments. Note that the segmentation is done using bsnt
#                       emg_filtered=emg_filtered,
                        emg_filtered=emg_signal,
                        pulses=segments,
                        path="./figures/" + file_name[1] + "_segments",
                        show=show_intermediate)

    show_experiment_info(data, segments_number, time)

    power_time, power_means, start_power_means, mid_power_means, end_power_means = calculate_muscle_activities(activity_starts,  activity_ends, emg_filtered, time, checked_flexes, show_fatigue)

    p.plot_emg_power_means_figure(time=time, emg_data=emg_filtered,
                                  power_mean_time=power_time, power_means=power_means,
                                  path=create_fatigue_figure_path(file_name),
                                  show=show_fatigue, regression_line=show_fatigue)
    
    p.plot_emg_power_means_detials(time=time, emg_data=emg_filtered,
                                 power_mean_time=power_time, 
                                 start_power_means=start_power_means,
                                 mid_power_means=mid_power_means,
                                 end_power_means=end_power_means,
                                 regression_line=show_fatigue)
    
    
def process_csv(data, subject, channel, filtering_config=None, segments_number=1, segment_checked_number=1,
                checked_flexes=None,
                show_intermediate=False, show_fatigue=True):
    # Filtering and cleaning up emg data

    emg = 'emg' + str(channel)
    data = data[[emg, 'time']]

    # Fix baseline shift if 'filter_config.fix_baseline_shift' is True
    emg_signal = remove_mean(data[emg], data.time, filtering_config)

    # In case we want to process a given part of the whole data file
    emg_signal, time = split_data(emg_signal, data.time, segments_number, segment_checked_number, filtering_config)

    # Create filtered emg data & emg envelope
    emg_filtered, emg_envelope = clean_up_emg_signal(time, emg_signal, filtering_config, show=show_intermediate)

    activity_starts, activity_ends, smooth_signal, moving_average_time, moving_average_emg = get_signal_segments(
        emg_signal, time, filtering_config, show=False)
    
    add_segments_to_data_file(data, subject, channel, activity_starts, activity_ends)

    segments = get_segments(activity_starts, activity_ends)

    p.plot_emg_segments(time=time,
                        emg_raw=emg_signal,
#                       TODO: use the filtered signal when ploting the segments. Note that the segmentation is done using bsnt
#                       emg_filtered=emg_filtered,
                        emg_filtered=emg_signal,
                        pulses=segments,
                        path="./figures/" + emg + "_segments",
                        show=show_intermediate)

    show_experiment_info(data, segments_number, time)

    power_time, power_means, start_power_means, mid_power_means, end_power_means = calculate_muscle_activities(activity_starts,  activity_ends, emg_filtered, time, checked_flexes, show_fatigue)

    p.plot_emg_power_means_figure(time=time, emg_data=emg_filtered,
                                  power_mean_time=power_time, power_means=power_means,
                                  path=create_fatigue_figure_path(emg),
                                  show=show_fatigue, regression_line=show_fatigue)
    
    p.plot_emg_power_means_detials(time=time, emg_data=emg_filtered,
                                 power_mean_time=power_time, 
                                 start_power_means=start_power_means,
                                 mid_power_means=mid_power_means,
                                 end_power_means=end_power_means,
                                 regression_line=show_fatigue)
    

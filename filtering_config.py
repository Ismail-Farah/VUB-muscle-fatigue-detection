# A class that contains the most filtering configurations
import scipy as sp

# Constants
BANDPASS = 'bandpass'
LOWPASS = 'lowpass'


class FilteringConfigurations:

    def __init__(self,
                 window_size,
                 pre_ignore=0,
                 post_ignore=0,
                 fix_baseline_shift=False,
                 high_band=20,
                 low_band=400,
                 low_pass=10,
                 sampling_frequency=1000,
                 threshold_level=10):
        self.window_size = window_size
        self.pre_ignore = pre_ignore
        self.post_ignore = post_ignore
        self.fix_baseline_shift = fix_baseline_shift
        self.high_band = high_band
        self.low_band = low_band
        self.low_pass = low_pass
        self.sampling_frequency = sampling_frequency
        self.threshold_level = threshold_level


def create_bandpass_filter(filter_config: object, btype: str = BANDPASS) -> tuple:
    # Create bandpass filter for EMG
    # Normalise cut-off frequencies to sampling frequency
    high = filter_config.high_band / (filter_config.sampling_frequency / 2)
    low = filter_config.low_band / (filter_config.sampling_frequency / 2)
    bandpass = [high, low]
    b, a = sp.signal.butter(4, bandpass, btype=btype, output='ba')

    return a, b


def create_low_pass_filter(filter_config):
    # create low-pass filter and apply to rectified signal to get EMG envelope
    low_pass = filter_config.low_pass / (filter_config.sampling_frequency / 2)
    b2, a2 = sp.signal.butter(4, low_pass, btype=LOWPASS)

    return a2, b2, low_pass

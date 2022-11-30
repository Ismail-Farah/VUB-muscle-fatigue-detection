import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import mplcursors


def plot_emg_segments(time=None,
                      emg_raw=None,
                      emg_filtered=None,
                      pulses=None,
                      path=None,
                      show=False):
    """Create a summary plot from the output of signals.emg.emg.
    Parameters
    ----------
    time : array
        Signal time axis reference (mille seconds).
    emg_raw : array
        Raw EMG signal.
    emg_filtered : array
        Filtered EMG signal.
    pulses : array
        Indices of EMG pulses.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot.
    """

    fig = plt.figure()
    fig.suptitle('EMG Signal')

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    # raw signal
    ax1.plot(time, emg_raw, linewidth=2.5, label='Raw signal')

    ax1.set_ylabel('a.u')
    ax1.grid()

    ymin = np.min(emg_filtered)
    ymax = np.max(emg_filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    ax2.plot(time, emg_filtered, linewidth=2.5, label='Filtered signal')
    ax2.vlines(pulses, ymin, ymax,
               color='r',
               linewidth=2.5,
               label='Segments')

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('a.u')
    ax2.grid()
    fig.set_size_inches(w=9, h=7)
    # fig.tight_layout()

    if path is not None:
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')

    if show:
        fig.show()
    else:
        plt.close(fig)


def plot_emg_power_means_figure(time=None, emg_data=None,
                                power_mean_time=None, power_means=None,
                                path=None,
                                show=False, regression_line=True):
    """Create a summary plot from the output of emg signals..
    Parameters
    ----------
    time : array
        Signal time axis reference (mille seconds).
    power_mean_time : array
        Time for which the power mean was recorded
    emg_data : array
        Filtered EMG signal.
    power_means : array
        Indices of EMG power means.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot.
    regression_line : bool, optional
        If True, show a regression line
    """

    fig = plt.figure()
    fig.suptitle('EMG Filtered Signal')

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    ax1.plot(time, emg_data, linewidth=2.5, label='Filtered signal')
    ax1.set_ylabel('a.u')
    ax1.grid()

    ax2.plot(power_mean_time, power_means, linewidth=2.5, label='Muscle power mean')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power mean')
    ax2.legend()
    fig.set_size_inches(w=9, h=8)
    fig.tight_layout()

    if regression_line:
        show_regression_line(ax2, power_mean_time, power_means)

    if path is not None:
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')

    if show:
        fig.show()
    else:
        plt.close(fig)


def show_regression_line(ax2, power_mean_time, power_means, color="red"):
    # plot a regression line
    x = power_mean_time
    y = power_means
    # non-linear least squares to fit func to data
    p_opt, p_cov = curve_fit(func, x, y)
    # these are the fitted values a, b, c
    # a, b, c = p_opt
    m, b = p_opt
    # produce 100 values in the range we want to cover along x
    x_fit = np.linspace(min(x), max(x), 100)
    # compute fitted y values
    y_fit = [func(x, m, b) for x in x_fit]
    ax2.plot(x_fit, y_fit, c=color, label="regression line")


# linear function to fit the data
def func(x, m, b):
    return m * x + b


def show_annotation(sel):
    ind = int(sel.target.index) + 1
    frac = sel.target.index - ind
    x, y = sel.target
    sel.annotation.set_text(f'activation No:{ind} at time: {x:.0f}')


def plot_emg_power_means_detials(time=None, emg_data=None,
                                 power_mean_time=None, 
                                 start_power_means=None,
                                 mid_power_means=None,
                                 end_power_means=None,
                                 regression_line=None,
                                 path=None):
    fig = plt.figure()
    fig.suptitle('EMG Filtered Signal')

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)

    ax1.plot(time, emg_data, linewidth=2.5, label='Filtered signal')
    ax1.set_ylabel('a.u')
    ax1.legend()
    ax1.grid()

    ax2.plot(power_mean_time, start_power_means, linewidth=2.5, color='red', label='Starts')
    ax2.plot(power_mean_time, mid_power_means, linewidth=2.5, color='green', label='Mids')
    ax2.plot(power_mean_time, end_power_means, linewidth=2.5, color='orange', label='Ends')
    # show segment number that corresponds to a given point   
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", show_annotation)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power mean')
    ax2.grid()
    ax2.legend()
    fig.set_size_inches(w=9, h=10)
  

    if regression_line:
        ax3 = fig.add_subplot(313, sharex=ax1)
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Power mean')
        ax3.set_title('Power regression line')
        ax3.legend()
        ax3.grid()
        show_regression_line(ax3, power_mean_time, start_power_means, color="red")
        show_regression_line(ax3, power_mean_time, mid_power_means, color="green")
        show_regression_line(ax3, power_mean_time, end_power_means, color="orange")
     
    fig.tight_layout()

    if path is not None:
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'
        fig.savefig(path, dpi=200, bbox_inches='tight')

    if regression_line:
        fig.show()
    else:
        plt.close(fig)


def plot_comparison(time=None, emg_before=None, plot_title_before=None, emg_after=None, plot_title_after=None):
    # plot comparison of EMG data before amd after processing
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title(plot_title_before)
    plt.plot(time, emg_before)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=3)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2).set_title(plot_title_after)
    plt.plot(time, emg_after)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=3)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    plt.grid()
    fig.tight_layout()
    fig_name = 'figure ' + plot_title_before + ' and ' + plot_title_after + '.png'
    fig.set_size_inches(w=9, h=7)
    fig.savefig(fig_name, dpi=200, bbox_inches='tight')
    fig.show()


def plot_signal_transformations(emg, emg_rectified, emg_envelope, time, filter_configs, show=False):
    # plot graphs
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.subplot(1, 3, 1).set_title('Unfiltered,' + '\n' + 'unrectified EMG')
    plt.plot(time, emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.ylabel('EMG (a.u.)')
    plt.grid()
    plt.subplot(1, 3, 2)
    plt.subplot(1, 3, 2).set_title(
        'Filtered,' + '\n' + 'rectified EMG: ' + str(
            int(filter_configs.high_band * filter_configs.sampling_frequency)) + '-' + str(
            int(filter_configs.low_band * filter_configs.sampling_frequency)) + 'Hz')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.grid()
    plt.subplot(1, 3, 3)
    plt.subplot(1, 3, 3).set_title(
        'Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(
            int(filter_configs.low_pass * filter_configs.sampling_frequency)) + ' Hz')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (sec)')
    plt.grid()
    fig.set_size_inches(w=9, h=7)
    fig.tight_layout()
    fig_name = 'fig_' + str(int(filter_configs.low_pass * filter_configs.sampling_frequency)) + '.png'
    fig.set_size_inches(w=8, h=6)
    fig.savefig(fig_name)


    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_segments_with_moving_average(emg_data, time, moving_average_emg, moving_average_time,
                                      segments_starts, segments_ends, filtering_configs, show=False):
    # plot comparison of EMG filtered data vs moving average values
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.subplot(1, 2, 1).set_title('Filtered EMG Signal')
    plt.plot(time, emg_data)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (millisecond)')
    plt.ylabel('EMG (a.u.)')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.subplot(1, 2, 2) \
        .set_title('Moving average with window size: ' + str(filtering_configs.window_size))
    plt.plot(moving_average_time, moving_average_emg)
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.xlabel('Time (millisecond)')
    plt.ylabel('EMG averages (a.u.)')

    ax = plt.gca()
    y_min = np.min(moving_average_emg)
    y_max = np.max(moving_average_emg)
    alpha = 0.1 * (y_max - y_min)
    y_max += alpha
    y_min -= alpha
    ax.vlines(segments_starts, y_min, y_max,
              color='g',
              linewidth=2.5,
              label='Activity start')
    ax.vlines(segments_ends, y_min, y_max,
              color='r',
              linewidth=2.5,
              label='Activity end')
    plt.legend(["EMG signal ", "Activity start", "Activity end"])
    plt.grid()
    fig.set_size_inches(w=9, h=7)
    fig.tight_layout()
    fig_name = 'Moving average window figure.png'
    fig.set_size_inches(w=11, h=7)
    fig.savefig(fig_name, dpi=200, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)        

        
def plot_activation_detials(activation_number=None,
                            array=None,
                            array_start=None,
                            array_mid=None,
                            array_end=None,
                            power_start=None, frequencies_start=None, filtered_power_start=None,
                            power_mid=None, frequencies_mid=None, filtered_power_mid=None,
                            power_end=None, frequencies_end=None, filtered_power_end=None,
                            power_medians_list=None,
                            show=False):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(array)
    plt.plot(array_start, label="start of flex", color="red")
    plt.plot(array_mid, label="mid of flex", color="green")
    plt.plot(array_end, label="end of flex", color="orange")
    plt.legend(["Flex", "Start", "Mid", "End"])
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(frequencies_start, power_start, color='blue')
    plt.plot(frequencies_start, filtered_power_start, color='red')
    plt.plot(frequencies_mid, filtered_power_mid, color='green')
    plt.plot(frequencies_end, filtered_power_end, color='orange')
    plt.legend(["flex power", "start power", "mid power", "end power"])
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.subplot(1, 3, 3).set_title("Median Frequencies")
    plt.ylabel("Power (a. u.)")
    plt.xlabel("Position relative to the flex")
    plt.scatter(x=[0.2, 0.5, 0.8], y=power_medians_list, marker='*', c='r', s=40)
    plt.plot([0.2, 0.5, 0.8], power_medians_list, color='blue')
    # plt.axhline(y = np.median(power_medians_list), color = 'g', linestyle = '-')
    plt.grid()
    
    fig.set_size_inches(w=9, h=5)
    fig.suptitle('Activation '+ str(activation_number) + ' details')
    
    if show :
        fig.savefig("Fig_contraction no: " + str(activation_number))
        plt.show()
    else:
        plt.close(fig)
        
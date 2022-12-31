import signal_processor as sgp
import filtering_config as fc

# Prepare fitering configurations
fc1 = fc.FilteringConfigurations(window_size=40,
                                 pre_ignore=0,
                                 fix_baseline_shift=False,
                                 high_band=20,
                                 low_band=400)


def analyze(subject=None, channels=[]):
    if subject is None:
        raise FileNotFoundError("Please specify the subject to be analyze.")

    for channel in channels:
        print("############## Channel (" + str(channel) + ") analysis ##############")
        sgp.process_emg(data_file_name=subject + "_data/files/emg" + str(channel) + "_data", filtering_config=fc1,
                        checked_flexes=[0, 50, 100],
                        show_intermediate=False)
        print("############## End of channel (" + str(channel) + ") analysis ############## \n")
        print()


def analyze_csv(subject=None, file=None, channels=[]):
    if subject is None:
        raise FileNotFoundError("Please specify the subject to be analyze.")
    if file is None:
        raise FileNotFoundError("Please specify the file to be analyze (e.g. 'right_arm', 'right_leg'...).")

    data = sgp.import_csv_data(subject + "_data/" + file , separator="\t")

    for channel in channels:
        print("############## Channel (" + str(channel) + ") analysis ##############")
        sgp.process_csv(data, subject, channel, filtering_config=fc1, segments_number=1,
                checked_flexes=[1,50,100],
                show_intermediate=False)
        print("############## End of channel (" + str(channel) + ") analysis ############## \n")
        print()

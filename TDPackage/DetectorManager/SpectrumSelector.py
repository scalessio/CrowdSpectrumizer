"""
Spectrum Selector Class: This class read chronological spectrum RawFFT data collected through the API and
store as numpy file the portion of the spectrum that we want to use to identify the the single transmissions.
This class takes as input the path where are stored the Raw FFT data process those data and store as numpy file
the portion selected.

This class is used by the process_fft_sweeps.py which we select the spectrum bands that we want to inspect and store the
spectrum data.

When we store the npy file with the raw fft data we can use this file by the Detector class that will
inspect the number of transmissions using only the psd data.

"""
import numpy as np
import matplotlib.pyplot as plt
from Deployment.TDPackage.Utils.chDetUtils  import load_dataset_raw,load_json_file
from collections import Counter
from numba import jit
class SpectrumSelector():
    def __init__(self):
        # Own variables Spectrum Selector Class
        self.debug = False
        self.actual_psd_values = []
        self.actual_frequencies = []
        self.actual_timeStamp = []

    @staticmethod
    def filter_frequency_values(entire_spectrum_frequency, start_freq=80000000, end_freq=81000000):
        # return the vector of the selected frequenies
        filtered_frequency = filter(lambda entire_spectrum_frequency: entire_spectrum_frequency >= start_freq,
                                    entire_spectrum_frequency)
        filtered_frequency = filter(lambda filtered_frequency: filtered_frequency <= end_freq, filtered_frequency)
        return list(filtered_frequency)

    @staticmethod
    def filter_indx_frequency(entire_spectrum_frequency, start_freq=80000000, end_freq=81000000):
        # filter the index from the array of frequencies
        a = np.asarray(entire_spectrum_frequency)
        idx = np.where(np.logical_and(a >= start_freq, a <= end_freq))
        return idx[0]

    def Most_Common(self,lst):
        data = Counter(lst)
        return data.most_common(1)[0][0]

    @jit(forceobj = True)
    def func_to_gpu2(self, response):
        for sensorData in response:
            frequencyResolution = sensorData['SenConf']['FrequencyResolution']  # in Hz
            centerFrequency = sensorData['SenData']['CenterFreq']  # Take the Center Freq in Hz
            timeStamp = sensorData['SenTime']['TimeSecs']
            squaredMag = sensorData['SenData']['SquaredMag']  # Take the chunk in the dict
            squaredMagLength = len(squaredMag)  # 214
            startFrequency = centerFrequency - ((squaredMagLength / 2) * frequencyResolution)
            endFrequency = centerFrequency + ((squaredMagLength / 2) * frequencyResolution)
            for i in range(squaredMagLength):  # Append Each 214 bin on a huge big vector
                self.actual_psd_values.append(squaredMag[i])
                self.actual_frequencies.append((startFrequency + i * frequencyResolution))
                self.actual_timeStamp.append(timeStamp)


    def func_to_gpu(self,fileNames):
        for fileName in fileNames:
            print("Read.. " + fileName)
            response = load_json_file(fileName)
            self.func_to_gpu2(response)



    def test_function_namba(self, directory, start_freq, end_freq, num_files=10):
        fileNames = load_dataset_raw(directory)  # Load in chronological order json file
        fileNames = fileNames[:num_files]
        datatype = "float32"
        if self.debug:
            plt.plot(range(len(self.actual_frequencies)), self.actual_frequencies)
            plt.show()
        # run this in the GPUs
        self.func_to_gpu(fileNames)
        # Looking where are the actual frequencies from the RTL-SDR and take the start.
        index_start = np.where(self.actual_frequencies == np.min(self.actual_frequencies))
        index_start = index_start[0]
        index_start = index_start[1:]  # Filter out firsts sweeps

        if self.debug:
            # Check the data integrity: each entire sweeps (from 24-1.7GHz) would have
            # the same number of samples.
            print(np.diff(index_start))

        # Build the spectrogram: Buld the frequency array.
        entire_spectrum_frequency = self.actual_frequencies[index_start[0]:index_start[1] - 1]
        if self.debug:
            print(" The entire sweep frequency array. ", len(entire_spectrum_frequency))

        time_segments = []
        time_stamps = []
        for i in range(0, len(index_start) - 1):
            time_segments.append(self.actual_psd_values[index_start[i]:index_start[i + 1] - 1])
            time_stamps.append(self.actual_timeStamp[index_start[i]])

        count_frq = []
        for i in range(0, len(time_segments)):
            # print("Print len timesegments ", len(time_segments))
            # print("Print len timeseg ", len(time_segments[i]))
            count_frq.append(len(time_segments[i]))

        mc = self.Most_Common(count_frq)
        # mode_of_timesegs = scipy.stats.mode(count_frq)

        # Staked up each single array of entire spectrum
        entire_data = np.zeros(mc, dtype=datatype)
        # Create the spectrogram with the Time segments, here handle missed chunks for each single sweeps
        for i in range(0, len(time_segments)):
            try:
                entire_data = np.vstack((entire_data, time_segments[i]))
            except Exception as error:
                print(error)
                pass

        # Now that we have the full spectrogram 20-1.7GHz Filter the portion of spectrum that we want to save.
        filtered_frequencies = self.filter_indx_frequency(entire_spectrum_frequency, start_freq=start_freq,
                                                          end_freq=end_freq)
        entire_data = entire_data[:, filtered_frequencies]
        entire_data = np.delete(entire_data, 0, 0)
        print("[Server]:Test Spectrum shape ", entire_data.shape)
        return entire_data


    def build_spc_portion(self, directory, start_freq, end_freq, num_files=10):
        fileNames = load_dataset_raw(directory)  # Load in chronological order json file
        fileNames = fileNames[:num_files]
        datatype = "float32"
        if self.debug:
            print(fileNames)
        for fileName in fileNames:
            print("Read.. " + fileName)
            response = load_json_file(fileName)
            for sensorData in response:
                frequencyResolution = sensorData['SenConf']['FrequencyResolution']  # in Hz
                centerFrequency = sensorData['SenData']['CenterFreq']  # Take the Center Freq in Hz
                timeStamp = sensorData['SenTime']['TimeSecs']
                squaredMag = sensorData['SenData']['SquaredMag'] # Take the chunk in the dict
                squaredMagLength = len(squaredMag)  # 214

                startFrequency = centerFrequency - ((squaredMagLength / 2) * frequencyResolution)


                endFrequency = centerFrequency + ((squaredMagLength / 2) * frequencyResolution)
                for i in range(squaredMagLength): # Append Each 214 bin on a huge big vector
                    self.actual_psd_values.append(squaredMag[i])
                    self.actual_frequencies.append((startFrequency + i * frequencyResolution))
                    self.actual_timeStamp.append(timeStamp)

        if self.debug:
            plt.plot(range(len(self.actual_frequencies)), self.actual_frequencies)
            plt.show()

        # Looking where are the actual frequencies from the RTL-SDR and take the start.
        index_start = np.where(self.actual_frequencies == np.min(self.actual_frequencies))
        index_start = index_start[0]
        index_start = index_start[1:]  # Filter out firsts sweeps

        if self.debug:
            # Check the data integrity: each entire sweeps (from 24-1.7GHz) would have
            # the same number of samples.
            print(np.diff(index_start))

        # Build the spectrogram: Buld the frequency array.
        entire_spectrum_frequency = self.actual_frequencies[index_start[0]:index_start[1] - 1]
        if self.debug:
            print(" The entire sweep frequency array. ", len(entire_spectrum_frequency))

        time_segments = []
        time_stamps = []
        for i in range(0, len(index_start) - 1):
            time_segments.append(self.actual_psd_values[index_start[i]:index_start[i + 1] - 1])
            time_stamps.append(self.actual_timeStamp[index_start[i]])

        count_frq = []
        for i in range(0,len(time_segments)):
            #print("Print len timesegments ", len(time_segments))
            #print("Print len timeseg ", len(time_segments[i]))
            count_frq.append(len(time_segments[i]))

        mc = self.Most_Common(count_frq)
        # mode_of_timesegs = scipy.stats.mode(count_frq)

        # Staked up each single array of entire spectrum
        entire_data = np.zeros(mc, dtype=datatype)
        # Create the spectrogram with the Time segments, here handle missed chunks for each single sweeps
        for i in range(0, len(time_segments)):
            try:
                entire_data = np.vstack((entire_data, time_segments[i]))
            except Exception as error:
                print(error)
                pass

        # Now that we have the full spectrogram 20-1.7GHz Filter the portion of spectrum that we want to save.
        filtered_frequencies = self.filter_indx_frequency(entire_spectrum_frequency, start_freq=start_freq,
                                                          end_freq=end_freq)
        entire_data = entire_data[:, filtered_frequencies]
        entire_data = np.delete(entire_data, 0, 0)
        print("[Server]:Test Spectrum shape ", entire_data.shape)
        return entire_data

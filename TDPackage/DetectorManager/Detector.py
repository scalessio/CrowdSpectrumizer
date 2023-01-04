import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ChannelDetector:

    def __init__(self, noiseThres, cv, smoothing, k, widthApplicable, borders=None):


        self.spectrum_path = None
        # Own variables detected Transmissions
        self.channels_detected = None  # This is the final array that contains the start and end bin.
        self.peaks_detected = None
        self.data = None
        self.noiseThres = noiseThres
        self.cv = cv
        self.smoothing = smoothing
        self.k = k
        self.widthApplicable = widthApplicable
        self.borders = borders

    def plt_the_image(self, data_plot, label):
        fig, ax1 = plt.subplots(figsize=(14, 8))
        img1 = ax1.imshow(data_plot, aspect='auto', cmap=cm.jet)
        # fig.colorbar(img1, ax=ax1,label='Db')
        ax1.set_xlabel("Frequency Bins", fontsize=32)
        ax1.set_ylabel("Time Segments", fontsize=32)
        # ax1.set_title("%s" % label, fontsize=28)

        if self.channels_detected is not None:
            import matplotlib.patches as patches
            imageLower = len(data_plot) - 1
            for channel in self.channels_detected:
                # if channel[0]<=470:
                print(channel)
                channelStart = channel[0]  # -700
                channelEnd = channel[1]  # -700
                rect = patches.Rectangle((channelStart, 0), channelEnd - channelStart, imageLower, linewidth=3.8,
                                         linestyle='--',
                                         edgecolor='black', facecolor='none')
                ax1.add_patch(rect)

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(32)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(32)
            # specify integer or one of preset strings, e.g.
            # tick.label.set_fontsize('x-small')
            # tick.label.set_rotation('vertical')
        plt.tight_layout()
        # plt.savefig('/home/alessio/%s.pdf'%label, dpi=500)
        plt.show()

    def get_channel_detected(self):
        return self.channels_detected

    def load_data(self, filename):
        self.data = np.load(filename, allow_pickle=True).astype(float)
        print("Data Loaded", self.data.shape)

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def find_peaks_and_edges(self, averaged_data, channelStart, channelEnd, peakThres, channels):
        channel_values_abs = averaged_data[channelStart:(channelEnd + 1)]

        # compute the coefficient of variation (CV)
        channel_mean_value = np.mean(channel_values_abs)
        channel_std_value = np.std(channel_values_abs)
        cv = channel_std_value / channel_mean_value

        # CV defined only for distributions with a positive mean values        
        if channel_mean_value > 0 and cv > self.cv:
            channel_smoothed = None
            if (self.smoothing):
                channel_smoothed = self.smooth(channel_values_abs, 8)
            else:
                channel_smoothed = channel_values_abs

            peaks = None
            if (self.widthApplicable):
                k = self.k
                peaks, _ = find_peaks(channel_smoothed,
                                      distance=10,
                                      width=4,
                                      prominence=np.mean(channel_smoothed) + k * np.std(channel_smoothed)
                                      )
            else:
                peaks, _ = find_peaks(channel_smoothed, distance=10)

            debu = False
            if debu:
                fig, ax1 = plt.subplots()
                ax1.plot(channel_values_abs)
                ax1.plot(peaks, channel_values_abs[peaks], "x", label='Peak')
                ax1.set_xlabel("%d" % channelStart)
                ax1.set_ylabel("Time Segments")
                plt.show()

            # todo: Put function find_peak_edges here, it's the same code. right?
            for peak in peaks:
                value_peak = channel_smoothed[peak]

                peak_freq_start = None
                peak_freq_end = None

                search = peak
                while search - 1 >= 0:
                    search -= 1
                    value_i = channel_smoothed[search]
                    log_value = 20 * np.log10(value_peak / value_i)
                    if log_value > peakThres:
                        peak_freq_start = search
                        break

                search = peak
                channel_smoothed_length = len(channel_smoothed)
                while search + 1 <= (channel_smoothed_length - 1):
                    search += 1
                    value_i = channel_smoothed[search]
                    log_value = 20 * np.log10(value_peak / value_i)
                    if log_value > peakThres:
                        peak_freq_end = search
                        break

                if peak_freq_start is None or peak_freq_end is None:
                    print("Skipping peak in " + str(channelStart + peak))
                    pass
                else:
                    channels.append([channelStart + peak_freq_start, channelStart + peak_freq_end, channelStart + peak])

            debu = False
            if debu:
                fig, ax1 = plt.subplots()
                ax1.plot(channel_smoothed)
                ax1.set_xlabel("%d" % channelStart)
                ax1.plot(peaks, channel_smoothed[peaks], "x", label='Peak')
                ax1.set_ylabel("Time Segments")
                plt.show()

            if len(peaks) > 0:
                pass
                # prominences = properties['prominences']

                # print("Mean value: " + str(channel_mean_value))
                # contour_heights = channel_smoothed[peaks] - prominences
                # plt.plot(channel_smoothed)
                # plt.plot(peaks, channel_smoothed[peaks], "x")
                # # plt.vlines(x=peaks, ymin=contour_heights, ymax=channel_smoothed[peaks])
                # aaaa = np.power(10, (20 * np.log10(channel_smoothed[peaks[0]]) - peakThres) / 20)
                # plt.hlines(y= aaaa, xmin=peak_freq_start, xmax=peak_freq_end, color='C2')
                # plt.vlines(x=[peak_freq_start, peak_freq_end], ymin=[0,0], ymax=[aaaa, aaaa], color='C3')
                # plt.show()
            else:
                channels.append([channelStart, channelEnd])
        else:
            channels.append([channelStart, channelEnd])

    def tx_detection_funct(self, noiseDb, windowLength=5, noiseThres=7, peakThres=1):
        # convert to voltage absolute
        spec_data = np.power(10, self.data / 20)
        averaged_data = np.mean(spec_data, axis=0)
        # actual channel identification
        in_channel = False
        channelStart = -1
        channelEnd = -1
        channels = []
        length = len(averaged_data)
        i = 0
        while i < length:
            if i + windowLength > length:
                break
            windowStart = i
            windowEnd = i + windowLength - 1
            window_values_abs = averaged_data[windowStart: (windowEnd + 1)]
            window_values_abs_mean = np.mean(window_values_abs)
            # convert the mean of absolute voltage values to db
            diff_db = 20 * np.log10(window_values_abs_mean) - noiseDb
            if diff_db > noiseThres:
                if not in_channel:
                    channelStart = windowStart
                channelEnd = windowEnd
                in_channel = True
                i += 1
            else:
                if in_channel:
                    if channelEnd - channelStart >= 10:
                        self.find_peaks_and_edges(averaged_data, channelStart, channelEnd, peakThres,
                                                  channels)

                    in_channel = False
                    i += windowLength
                else:
                    i += 1

        if in_channel:
            if channelEnd - channelStart >= 10:
                self.find_peaks_and_edges(averaged_data, channelStart, channelEnd, peakThres, channels)

        start = [x[0] for x in channels]
        end = [x[1] for x in channels]
        pk = [x[2] for x in channels if len(x) == 3]
        deb = False
        if deb:
            plt.figure(figsize=(8, 8))
            averaged_data = 20 * np.log10(averaged_data)
            plt.plot(averaged_data)
            plt.plot(pk, averaged_data[pk], "x", label='Peak', markersize=12)
            plt.plot(start, averaged_data[start], "*", label='Start Transmission', markersize=12)
            plt.plot(end, averaged_data[end], "o", label='End Transmission', markersize=12)
            # plt.plot(self.channels_detected[0,:], sig[self.channels_detected[0,:]], "o")
            # plt.plot(self.channels_detected[1,:], sig[self.channels_detected[1,:]], "o")
            plt.axhline(noiseDb, xmin=0.0, label="Noise Floor", color='orange')
            plt.legend(loc='best', fontsize=24)
            ##plt.savefig('/home/alessio/pk_edge_detection.pdf', dpi=500)
            plt.show()

        if len(channels) > 0:
            self.channels_detected = np.array([start, end]).T
            self.peaks_detected = np.array([pk])
        else:
            self.channels_detected = None

    def detect_transmissions(self, filename, type="N", noiseDb=None, peakThres=3):
        self.load_data(filename)
        # self.plt_the_image(self.data[:,:], "Original Spectrum Data")
        self.tx_detection_funct(noiseDb, noiseThres=self.noiseThres, peakThres=peakThres)
        # self.plt_the_image(self.data[:60,:], "GSM Transmissions \n [924-929] MHz")
        if self.channels_detected is not None:
            print("Transmissions Detected")
            return self.channels_detected
        else:
            print("NO Transmissions detected")
            return self.channels_detected

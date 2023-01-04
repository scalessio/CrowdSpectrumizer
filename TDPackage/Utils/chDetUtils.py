import os
import re
import numpy as np
import json
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd


def store_metadata(basepath, label, transmission_n, sensor_id, sensor_name, freq_res, time_res,
                   timeBegin, timeEnd, SNR, start_transmission, end_transmission):
    # print("Save Medata..")
    file_number = transmission_n
    metadata_filename = basepath + "%s_%d.csv" % (label, file_number)
    columns = ['id_sensor', 'name_sensor', 'freq_reso', 'time_res', 'begin_time', 'end_time', 'begin_freq',
               'end_freq', 'SNR']
    metadata = [sensor_id, sensor_name, freq_res, time_res,
                timeBegin, timeEnd, start_transmission, end_transmission, SNR]
    df_metadata = pd.DataFrame(list(metadata), index=columns).T
    # print(df_metadata)
    df_metadata.to_csv(metadata_filename, index=False)


def extract_bins_from_matrix(matrix, num_of_bins):
    result = None
    length = np.shape(matrix)[1]
    if (length <= num_of_bins):
        return matrix
    j = 0
    max = float('-inf')
    while j + num_of_bins <= length:
        x = matrix[:, j: j + num_of_bins]
        sum = x.sum()
        if (sum >= max):
            max = sum
            result = x
        j += 1
    return result


def compute_snr_equally(dta, noise, label):
    lenght = 0
    if label == 'tetra':
        lenght = 5
    elif label == 'fm':
        lenght = 10
    elif label == 'gsm':
        lenght = 10
    else:
        lenght = 200
    dta = extract_bins_from_matrix(dta, lenght)

    dta = dta - noise
    # Transform in Absolute Scale And Compute the Mean of Transmission as for SNR.
    # Restore Back in dB scale.
    SNR_abs = np.power(10, dta / 20)
    SNR_abs = np.mean(SNR_abs)
    SNR = np.log10(SNR_abs * 20)
    return SNR


def plt_ch_corr_confusion_matrix(confusion_matrix):
    import seaborn as sn
    import pandas as pd

    df_cm = pd.DataFrame(confusion_matrix,
                         index=["Ch. " + str(i + 1) for i in range(0, len(confusion_matrix))],
                         columns=["Ch. " + str(i + 1) for i in range(0, len(confusion_matrix))])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)

    plt.show()


def load_json_file(fileName):
    with open(fileName) as json_file:
        return json.load(json_file)


def sort_nicely(l):
    # ref: https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c.replace("_", "")) for c in re.split('([0-9]+)', key)]
    l.sort(key=alphanum_key)
    return l


def load_dataset_raw(directories):
    files = []
    for dir in directories:
        for file in os.listdir(dir):
            files.append(os.path.join(dir, file))
    match = [f for f in files if not f.__contains__(".csv")]
    return sort_nicely(match)


def define_noise_level(fileNames):
    lista = []
    response = load_json_file(fileNames)  # filename contains a full sweep
    for sensorData in response:
        squaredMag = sensorData['SenData']['SquaredMag']  # 214
        # compute the mean on each chunks in absolute scale
        lista.append(np.mean([np.power(10, x / 20) for x in squaredMag]))

    # Detect the chunk with the lowest AVG and Create the min_bin_vector
    result = np.min(lista)  # compute the min of the averaged means
    min_index = lista.index(result)  # take the index of the averaged means
    min_bin = response[min_index]  # extract the chunk with the lowest mean
    # Transform the chunk in absolut scale
    min_bin_values = [np.power(10, x / 20) for x in min_bin['SenData']['SquaredMag']]

    min_bin_mean = np.mean(min_bin_values)  # compute the average of the chunk
    min_bin_stdv = np.std(min_bin_values)  # compute the standard deviation of the chunk

    # Std is applied on absolute scale
    noise_abs = min_bin_mean + 3 * min_bin_stdv
    noise_db = 20 * np.log10(noise_abs)
    return noise_abs, noise_db


def plt_the_image(data_plot, label):
    fig, ax1 = plt.subplots()
    img1 = ax1.imshow(data_plot, aspect='auto', cmap=cm.jet)
    fig.colorbar(img1, ax=ax1)
    ax1.set_xlabel("Bin", fontsize=15)
    ax1.set_ylabel("Time Segments", fontsize=15)
    ax1.set_title("%s" % label, fontsize=18)
    plt.grid()
    plt.show()


def plot_trace(dta_new, f_name='Trace', country='Country', snr_value=0.0, savepath=None, save=False):
    fig, ax1 = plt.subplots()
    img1 = ax1.imshow(dta_new, aspect='auto', cmap=cm.jet)
    fig.colorbar(img1, ax=ax1)
    ax1.set_title('%s %s - SNR %.2f' % (country, f_name, snr_value), fontsize=13)
    ax1.set_xlabel("Bin")
    ax1.set_ylabel("Time Segments")
    if save:
        plt.savefig(savepath.replace("npy", "jpeg"), dpi=None, facecolor='w', edgecolor='w',
                    papertype=None, format=None, quality=99,
                    transparent=False, bbox_inches=None,
                    frameon=None, metadata=None)

    plt.show()


def plt_trace_and_cdf(dta, start, end, j):
    mean = np.mean(dta, axis=0)
    normalized_data = (mean - np.mean(mean)) / np.std(mean)
    x, y = sorted(mean), np.arange(len(mean)) / len(mean)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    img1 = ax[0].imshow(dta, aspect='auto', cmap=cm.jet)
    fig.colorbar(img1, ax=ax[0])

    ax[0].set_xlabel('Bin')
    ax[0].set_ylabel('Time Segment')
    ax[0].set_title('Transmission %d \n (%.2f - %.2f) MHz' % (j, start, end))

    ax[1].plot(x, y)
    ax[1].set_ylabel('p')
    ax[1].set_xlabel('Power Value')
    ax[1].set_title('CDF')
    plt.xlim([-70, -20])
    plt.show()


def ret_avgstd_bins_over_time(entire, label, std=False, plot_avg=False):
    # is the same of the other but here we can choose to plot or not
    fig, ax = plt.subplots(1, 1)
    mean_bins = np.mean(entire, axis=0)
    ax.plot(mean_bins, label="%s avg" % label, lw=1, color='red')
    ax.set_xlabel("Bins", fontsize=15)

    if std:
        std_bins = np.std(entire, axis=0)
        ax.plot(std_bins, label="%s Std" % label, lw=1, color='blue')
        ax.set_ylabel("AVG and STD", fontsize=15)
        ax.set_title("AVG and STD Over Time %s" % label, fontsize=15)
        ax.legend()
        if plot_avg:
            plt.show()
        return mean_bins, std_bins
    else:
        ax.set_ylabel("AVG", fontsize=15)
        ax.set_title("Bin AVG over Time %s" % label, fontsize=15)
        if plot_avg:
            plt.show()
        return mean_bins


def store_transmissions_data(store_txs_path, transmissions, tx_test, store_single_tx, noise_db, sensor_id, snsname):
    try:
        os.makedirs(store_txs_path)
    except Exception as error:
        print(error)
    finally:
        os.system('rm -rf %s*' % store_txs_path)
        np.save(store_txs_path + "%s_results.npy" % 'FindOpenSSL.cmake', transmissions)

    # Store single transmissions
    try:
        data = tx_test
        f_start = 20000000
        plot = True
        store_meta = True
        if store_single_tx:
            for i in range(len(transmissions)):
                channel = transmissions[i]
                dta = data[:, np.round(channel[0]).astype(int):(np.round(channel[1]).astype(int) + 1)]
                start = ((f_start + float(channel[0] * 1e4)) / 1e6)
                end = ((f_start + float(channel[1] * 1e4)) / 1e6)
                np.save(store_txs_path + "test_" + str(i), dta)
                if plot:
                    SNR = compute_snr_equally(dta, noise_db, 'test')
                    plot_trace(dta, f_name="(%.2f - %.2f) MHz " % (start, end),
                               country='Detected Transmission %d.\n' % (i + 1),
                               snr_value=SNR, savepath=None, save=False)
        if store_meta:
            k = 0
            for i in transmissions:
                dta = data[:, np.round(i[0]).astype(int):(np.round(i[1]).astype(int) + 1)]
                SNR = compute_snr_equally(dta, noise_db, 'FindOpenSSL.cmake')
                start = ((f_start + float(i[0] * 1e4)) / 1e6)
                end = ((f_start + float(i[1] * 1e4)) / 1e6)
                # print(start, end)
                if end < start:
                    print("ERROR")
                sensor_id = sensor_id
                freq_res = 10000
                time_res = 60
                timeBegin = 1000000
                timeEnd = 2000000
                store_metadata(store_txs_path, 'FindOpenSSL.cmake', k, sensor_id, snsname, freq_res, time_res,
                               timeBegin, timeEnd, SNR, start, end)
                k += 1
    except Exception as error:
        print(error)

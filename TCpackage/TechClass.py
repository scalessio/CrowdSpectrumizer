import time
import numpy as np
import tensorflow as tf
import math
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches, cm
from tsfresh.feature_extraction.feature_calculators import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from Deployment.TDPackage.DetectorManager.Detector import ChannelDetector


class TechnologyClassifClass:

    def __init__(self, f_start, idsensor, name_sensor):
        self.CHUNKSIZE = 450
        self.initial_feat_leng = 34
        self.data = []
        self.f_start = int(f_start)
        self.transmissions = []
        self.index = 0
        self.id_sns = idsensor
        self.name_sns = name_sensor
        self.labels_detected = []
        self.noisedb = 0
        self.first_time = True
        self.dic = {0: 'dab', 1: 'dvbt', 2: 'fm', 3: 'gsm', 4: 'lte', 5: 'tetra', 6: 'unkn'}
        self.datatype = "float32"
        self.key_tr_lab = 'unkn'
        self.list_entropy = []

        self.BASE_MODEL_PATH = os.getcwd() + '/Deployment/TCpackage/resources/'
        self.model_path = self.BASE_MODEL_PATH + 'save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/saved-model-110-0.97.hdf5'
        self.model_dir = self.BASE_MODEL_PATH + 'save-DL-models/LSTM_TrainWithAE/TrainAllSensorHop_DNNAE16_LSTM_mse_relu/'
        self.encoder_path = self.BASE_MODEL_PATH + 'save-DL-models/Autoencoder_DNN/TrainAllSensorHop_16_feat_mse_relu/saved-model-49-0.0002.hdf5'
        self.scaler_path = self.BASE_MODEL_PATH + 'scaler/_AE16_LSTM_Scaler_.save'  # /home/alessio/tmp/pycharm_project_679/TechClassification/DataTraining/_AE16_LSTM_Scaler_.save'
        self.detected_transmissions_path = None
        self.spectrum_path = None
        self.scaler = joblib.load(self.scaler_path)
        self.encoder, self.model = self.load_NNs()
        self.tx_label = []

    def load_dnn_encoder(self, activ_f):
        encoder = Sequential()
        encoder.add(Dense(units=64, activation=activ_f, input_shape=[33]))
        encoder.add(Dense(units=32, activation=activ_f))
        encoder.add(Dense(units=16, activation=activ_f))
        decoder = Sequential()
        decoder.add(Dense(units=16, activation=activ_f, input_shape=[16]))
        decoder.add(Dense(units=32, activation=activ_f))
        decoder.add(Dense(units=64, activation=activ_f))
        decoder.add(Dense(units=33))
        autoencoder = Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')  # optimizer=tf.keras.optimizers.Adam(lr=ln_rate)
        autoencoder.load_weights(self.encoder_path)
        encoder = autoencoder.layers[0]
        encoder.summary()
        return encoder

    def load_classifier(self, outputs, timesteps, features):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(timesteps, features), return_sequences=True))
        model.add(LSTM(16, activation='relu'))
        model.add(Dense(16, activation='softmax'))
        model.add(Dropout(0.001))
        model.add(Dense(outputs, activation='softmax'))  # todo: update with the newest model
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      metrics=['accuracy'])
        model.load_weights(self.model_path)
        return model

    def load_NNs(self):
        encod_type = 'DNN'
        latent_s = '16'
        activ_f = 'relu'
        timesteps = 16
        features = 1
        outputs = 6
        encoder = self.load_dnn_encoder(activ_f)
        model = self.load_classifier(outputs, timesteps, features)
        return encoder, model

    def extract_statitics(self, dta, id_sns):
        df = pd.DataFrame(dta)
        df_transf = pd.DataFrame()
        # df_transf['bw'] = df.apply(lambda x_vector : len(x_vector),axis=1)
        df_transf['abs_energy'] = df.apply(lambda x_vector: abs_energy(x_vector), axis=1)
        df_transf['absolute_sum_of_changes'] = df.apply(lambda x_vector: absolute_sum_of_changes(x_vector), axis=1)
        df_transf['benford_correlation'] = df.apply(lambda x_vector: benford_correlation(x_vector), axis=1)
        df_transf['cid_ce'] = df.apply(lambda x_vector: cid_ce(x_vector, True), axis=1)
        df_transf['count_above_mean'] = df.apply(lambda x_vector: count_above_mean(x_vector), axis=1)  # ------>>>
        df_transf['count_below_mean'] = df.apply(lambda x_vector: count_below_mean(x_vector), axis=1)  # ------>>>
        df_transf['first_location_of_maximum'] = df.apply(lambda x_vector: first_location_of_maximum(x_vector), axis=1)
        df_transf['first_location_of_minimum'] = df.apply(lambda x_vector: first_location_of_minimum(x_vector), axis=1)
        df_transf['has_duplicate'] = df.apply(lambda x_vector: has_duplicate(x_vector), axis=1)
        df_transf['has_duplicate_max'] = df.apply(lambda x_vector: has_duplicate_max(x_vector), axis=1)
        df_transf['has_duplicate_min'] = df.apply(lambda x_vector: has_duplicate_min(x_vector), axis=1)
        df_transf['kurtosis'] = df.apply(lambda x_vector: kurtosis(x_vector), axis=1)  # ------>>>
        df_transf['last_location_of_maximum'] = df.apply(lambda x_vector: last_location_of_maximum(x_vector), axis=1)
        df_transf['last_location_of_minimum'] = df.apply(lambda x_vector: last_location_of_minimum(x_vector), axis=1)
        df_transf['longest_strike_above_mean'] = df.apply(lambda x_vector: longest_strike_above_mean(x_vector), axis=1)
        df_transf['longest_strike_below_mean'] = df.apply(lambda x_vector: longest_strike_below_mean(x_vector), axis=1)
        df_transf['maximum'] = df.apply(lambda x_vector: maximum(x_vector), axis=1)
        df_transf['mean'] = df.apply(lambda x_vector: mean(x_vector), axis=1)  # ------>>>
        df_transf['mean_abs_change'] = df.apply(lambda x_vector: mean_abs_change(x_vector), axis=1)
        df_transf['mean_change'] = df.apply(lambda x_vector: mean_change(x_vector), axis=1)
        df_transf['mean_second_derivative_central'] = df.apply(
            lambda x_vector: mean_second_derivative_central(x_vector),
            axis=1)
        df_transf['median'] = df.apply(lambda x_vector: median(x_vector), axis=1)  # ------>>>
        df_transf['minimum'] = df.apply(lambda x_vector: minimum(x_vector), axis=1)
        df_transf['number_cwt_peaks'] = df.apply(lambda x_vector: number_cwt_peaks(x_vector, n=3), axis=1)
        df_transf['number_peaks'] = df.apply(lambda x_vector: number_peaks(x_vector, n=3), axis=1)
        df_transf['number_cwt_peaks'] = df.apply(lambda x_vector: number_cwt_peaks(x_vector, n=3), axis=1)
        df_transf['quantile'] = df.apply(lambda x_vector: quantile(x_vector, 0.5), axis=1)
        df_transf['root_mean_square'] = df.apply(lambda x_vector: root_mean_square(x_vector), axis=1)
        # df_transf['sample_entropy'] = df.apply(lambda x_vector : sample_entropy(x_vector),axis=1)
        df_transf['skewness'] = df.apply(lambda x_vector: skewness(x_vector), axis=1)  # ------>>>
        df_transf['standard_deviation'] = df.apply(lambda x_vector: standard_deviation(x_vector), axis=1)
        df_transf['sum_of_reoccurring_values'] = df.apply(lambda x_vector: sum_of_reoccurring_values(x_vector), axis=1)
        df_transf['sum_values'] = df.apply(lambda x_vector: sum_values(x_vector), axis=1)
        # df_transf['value_count'] = df.apply(lambda x_vector : value_count(x_vector,np.mean(x_vector)),axis=1)
        df_transf['variance'] = df.apply(lambda x_vector: variance(x_vector), axis=1)  # ------>>>
        df_transf['variation_coefficient'] = df.apply(lambda x_vector: variation_coefficient(x_vector),
                                                      axis=1)  # ------>>>
        df_transf['Id_sensor'] = id_sns

        return df_transf.values, list(df_transf.columns.values)

    def read_metafromdf(self, df):
        st_freq = df['begin_freq'].values[0]
        end_freq = df['end_freq'].values[0]
        SNR = df['SNR'].values[0]
        id_sns = df['id_sensor'].values[0]
        return SNR, st_freq, end_freq, id_sns

    def refine_df(self, datafr):
        datafr.replace([np.inf], np.nan, inplace=True)
        datafr.isnull().sum()
        datafr['skewness'] = datafr['skewness'].fillna(0)
        datafr['mean_second_derivative_central'] = datafr['mean_second_derivative_central'].fillna(0)
        datafr['kurtosis'] = datafr['kurtosis'].fillna(0)
        return datafr

    def calc_entropy(self, probs):
        my_sum = 0
        for p in probs:
            if p > 0:
                my_sum += p * math.log(p, 2)
        return - my_sum

    def scoreEntropyPred(self, test_Y_i_hat, treshold_alpha):
        for elem in test_Y_i_hat:
            h = self.calc_entropy(elem)
            self.list_entropy.append(h)

        entropy_avg = np.mean(self.list_entropy)
        if entropy_avg > treshold_alpha: # Unknown for prediction upper than threshold
            x = np.arange(len(test_Y_i_hat))
            y_label = np.full_like(x, 6)
            y_label = y_label.tolist()
            y_label = np.max(y_label)
        else:
            y_label = np.argmax(test_Y_i_hat.mean(axis=0))
        return y_label, entropy_avg

    def extract2MHz(self, dta, plot_f, SNR):
        print("Total TrX BW %d" % dta.shape[1])
        center_channel = round(dta.shape[1] / 2)
        if center_channel < 101:
            raise NameError('Error Small BW')
        else:
            if plot_f:
                fig, ax1 = plt.subplots(figsize=(14, 8))
                ax1.imshow(dta, aspect='auto', cmap=cm.jet)
                ax1.set_xlabel("Frequency Bins", fontsize=32)
                ax1.set_ylabel("Time Sweeps", fontsize=32)
                imageLower = len(dta) - 1
                plt.imshow(dta, aspect='auto', cmap=cm.jet)
                # plt.title('Transformed_%s_SNR_%s' % (f.split('/')[-1], str(SNR)))
                rect = patches.Rectangle((center_channel - 100, 0), (center_channel + 100) - (center_channel - 100),
                                         imageLower, linewidth=5.8,
                                         linestyle='--',
                                         edgecolor='black', facecolor='none')
                ax1.add_patch(rect)
                plt.show()
            dta = dta[:, (center_channel - 100):(center_channel + 100)]

        print("Transformed BW %d" % dta.shape[1])
        if plot_f:
            plt.imshow(dta, aspect='auto', cmap=cm.jet)
            # plt.title('Transformed_%s_SNR_%s' % (f.split('/')[-1], str(SNR)))
            plt.show()
        return dta

    def inference_data(self, X, encoder, model, snr):
        threshold_alpha = 0.7
        if snr > 3:
            X_test = self.scaler.transform(X)
            X_test_encode = encoder.predict(X_test)
            X_test_encode = np.reshape(X_test_encode, (-1, 16, 1))
            X_test = X_test_encode
            test_Y_i_hat = np.array(model.predict(X_test))
            # y_label, entropy_avg = self.compute_scores_alpha(test_Y_i_hat,threshold_alpha)
            y_label, entropy_avg = self.scoreEntropyPred(test_Y_i_hat, threshold_alpha)

        else:  # SNR lower than the one used for the training
            X_test = self.scaler.transform(X)
            X_test_encode = encoder.predict(X_test)
            X_test_encode = np.reshape(X_test_encode, (-1, 16, 1))
            X_test = X_test_encode
            test_Y_i_hat = np.array(model.predict(X_test))
            x = np.arange(len(test_Y_i_hat))
            y_label = np.full_like(x, 6).tolist()  # create a vector for output unknonw label
            y_label = np.max(y_label)
            for elem in test_Y_i_hat:
                h = self.calc_entropy(elem)
                self.list_entropy.append(h+threshold_alpha)
            entropy_avg = np.mean(self.list_entropy)
        return y_label, entropy_avg, test_Y_i_hat

    def loadAndPredict(self, file, SNR, data_tmp, DEF_NUM_TIMESEGMENTS):
        dta = np.load(self.detected_transmissions_path + file, allow_pickle=True)
        dta = dta[:DEF_NUM_TIMESEGMENTS, :]
        if dta.shape[1] >= 200:
            dta = self.extract2MHz(dta, False, SNR)
        res, cols = self.extract_statitics(dta, self.id_sns)
        data_tmp = np.vstack((data_tmp, res))
        data_tmp = np.delete(data_tmp, 0, 0)
        df = pd.DataFrame(data=data_tmp, columns=cols)
        df = self.refine_df(df)
        test_trace = df.values[:, :-1]  # remove the id_sensor
        y_label, entropy_avg, test_Y_i_hat = self.inference_data(test_trace, self.encoder, self.model, SNR)
        return y_label, entropy_avg, test_Y_i_hat

    def inference_trx_labels(self):
        DEF_NUM_TIMESEGMENTS = 50
        labels_results = []
        fnames = []
        prediction_pds = []
        list_labels_predicted = []
        entropies_avg = []
        snr_transmissions = []
        txs_starts = []
        txs_end = []

        count = 0
        # Classify each detected transmission (Stored file)
        for file in os.listdir(self.detected_transmissions_path):
            count = count + 1
            test_Y_i_hat = []
            y_label = 0
            entropy_avg = 0
            data_tmp = np.zeros(self.initial_feat_leng, dtype=self.datatype)

            if file.find("results") == -1 and file.endswith('.npy'):
                tx_test = np.load(self.detected_transmissions_path + file, allow_pickle=True)
                tx_test = tx_test[:DEF_NUM_TIMESEGMENTS, :]
                SNR, tx_st_freq, tx_end_freq, _ = self.read_metafromdf(
                    pd.read_csv(self.detected_transmissions_path + file.replace(".npy", ".csv"), header='infer'))

                if tx_st_freq > tx_end_freq:
                    print("error")
                print("Count %d - %f %f [MHz]" % (count, tx_st_freq, tx_end_freq))

                # Distinguish for untrained frequency
                if 420000000.0 * 1e6 <= tx_st_freq * 1e6 <= 780000000.0 * 1e6:
                    self.key_tr_lab = 'dvbt'
                elif 930000000.0 * 1e6 <= tx_st_freq * 1e6 < 980000000.0 * 1e6:
                    self.key_tr_lab = 'gsm'
                elif 80000000.0 <= tx_st_freq * 1e6 <= 130000000.0:
                    self.key_tr_lab = 'fm'
                elif 360000000.0 <= tx_st_freq * 1e6 <= 400000000.0:
                    self.key_tr_lab = 'tetra'
                elif 790000000.0 * 1e6 <= tx_st_freq * 1e6 <= 830000000.0 * 1e6:
                    self.key_tr_lab = 'lte'
                elif 180000000.0 <= tx_st_freq * 1e6 <= 240000000.:
                    self.key_tr_lab = 'dab'
                else:
                    self.key_tr_lab = 'unkn'

                if self.key_tr_lab == 'dab' and 120 <= tx_test.shape[1] <= 240 and SNR >= 1.70:
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'dvbt' and tx_test.shape[1] >= 400:
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'fm':
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'gsm' and 14 <= tx_test.shape[1] <= 35:
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'lte' and tx_test.shape[1] > 700:
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'tetra' and tx_test.shape[1] < 10:
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])
                if self.key_tr_lab == 'unkn':
                    y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp,
                                                                             DEF_NUM_TIMESEGMENTS)
                    labels_results.append([float(y_label), self.dic[y_label], tx_st_freq, tx_end_freq])

                fnames.append(file)
                prediction_pds.append(test_Y_i_hat)
                list_labels_predicted.append(y_label)  # append transmission label
                entropies_avg.append(entropy_avg)
                snr_transmissions.append(SNR)
                txs_starts.append(tx_st_freq)
                txs_end.append(tx_end_freq)

            df = pd.DataFrame(data=fnames, columns=['fname'])
            df['label'] = list_labels_predicted
            df['entropy_avg'] = entropies_avg
            df['snr'] = snr_transmissions
            df['startf_mhz'] = txs_starts
            df['endf_mhz'] = txs_end
            df.to_csv(self.detected_transmissions_path + 'df_result')  # store a file with the result

        return labels_results



    def experiment_framework_api(self, transmissions):
        entropy_AVG = []
        list_labels_predicted = []
        snr_transmissions = []
        fnames = []
        txs_starts = []
        txs_end = []
        list_yhat = []

        DEF_NUM_TIMESEGMENTS = 2
        print("# Detected Trx for the Full Spectr ", len(transmissions))
        list_of_detected_tx = transmissions
        # f_names = os.listdir(self.detected_transmissions_path)

        for file in os.listdir(self.detected_transmissions_path):
            data_tmp = np.zeros(self.initial_feat_leng, dtype=self.datatype)
            if file.find("results") == -1 and file.endswith('.npy'):
                SNR, tx_st_freq, tx_end_freq, _ = self.read_metafromdf(
                    pd.read_csv(self.detected_transmissions_path + file.replace(".npy", ".csv"), header='infer'))

                if tx_st_freq > tx_end_freq:
                    print("error")
                print(" %f %f [MHz]" % (tx_st_freq, tx_end_freq))

                if 420000000.0 + 54.32 * 1e6 <= tx_st_freq * 1e6 <= 780000000.0 + 54.32 * 1e6:
                    self.key_tr_lab = 'dvbt'
                elif 930000000.0 + 54.32 * 1e6 <= tx_st_freq * 1e6 < 980000000.0 + 54.32 * 1e6:
                    self.key_tr_lab = 'gsm'
                elif 80000000.0 <= tx_st_freq * 1e6 <= 130000000.0:
                    self.key_tr_lab = 'fm'
                elif 360000000.0 <= tx_st_freq * 1e6 <= 400000000.0:
                    self.key_tr_lab = 'tetra'
                elif 790000000. + 54.32 * 1e6 <= tx_st_freq * 1e6 <= 830000000.0 + 54.32 * 1e6:
                    self.key_tr_lab = 'lte'
                elif 180000000.0 <= tx_st_freq * 1e6 <= 240000000.:
                    self.key_tr_lab = 'dab'
                else:
                    self.key_tr_lab = 'unkn'

                y_label, entropy_avg, test_Y_i_hat = self.loadAndPredict(file, SNR, data_tmp, DEF_NUM_TIMESEGMENTS)
                list_yhat.append(test_Y_i_hat)
                fnames.append(file)
                list_labels_predicted.append(y_label)  # append transmission label
                entropy_AVG.append(entropy_avg)
                snr_transmissions.append(SNR)
                txs_starts.append(tx_st_freq)
                txs_end.append(tx_end_freq)
            # else:
            #    print('No npy transmissions file founded')
        df = pd.DataFrame(data=fnames, columns=['fname'])
        df['label'] = list_labels_predicted
        df['entropy_avg'] = entropy_AVG
        df['list_yhat'] = list_yhat
        df['snr'] = snr_transmissions
        df['startf_mhz'] = txs_starts
        df['endf_mhz'] = txs_end
        df.to_csv('./df_framework_experiments_%s' % self.name_sns)
        np.save('./FrameworkExp_transmissions_snr_transmissions_%s' % self.name_sns, np.array(snr_transmissions))
        np.save('./FrameworkExp_transmissions_entropies_%s' % self.name_sns, np.array(entropy_AVG))
        np.save('./FrameworkExp_transmissions_detected_%s' % self.name_sns, np.array(list_of_detected_tx))
        np.save('./FrameworkExp_transmissions_labeled_%s' % self.name_sns, np.array(list_labels_predicted))

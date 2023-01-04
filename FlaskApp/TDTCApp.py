#!flask/bin/python
import pickle
import sys
import os
import time
import numpy as np
from flask import Flask, jsonify, request, abort, json
sys.path.insert(0, '../Deployment/')
sys.path.insert(0, '../Deployment/TCpackage/')
sys.path.insert(0, '../Deployment/TDpackage/')
from TCpackage.TechClass import TechnologyClassifClass
from TDPackage.DetectorManager.Detector import ChannelDetector
from TDPackage.DetectorManager.SpectrumSelector import SpectrumSelector
from TDPackage.Utils.chDetUtils import define_noise_level, store_transmissions_data
from Utils.PredictionInfo import PredictionInformation

info_prediction = {
    'sensorid': '0001',
    'numbTx': '3',
    'statistics': [[2.0, 0.5, 'fm'], [2.0, 0.5, 'fm'], [2.0, 0.5, 'fm']],
}
app = Flask(__name__)
DEF_BASE_SERVICE_PATH = '/services/api/v1.0'
DEF_API_EXPERT_USER = DEF_BASE_SERVICE_PATH + '/usertc'
DEF_BASE_PATH = "./Deployment/FlaskApp"

"""
The Flask app exposes the API to predict the spectrum as input
"""


def serviceProcessRawFFTData(n_response_files=50):
    """
    This service is used to implement the extraction layer process the raw data from the sensor.
    - It consumes from kafka the messages from the sensor and process all the data.(require Pyspark)
    - We implement the version without using the cluster Kafka through the spectrum selector
    The Raw json files are stored in RAW.
    :return:
    """
    # process_data_from_kafka() # Call The extraction Layer
    skip_rawfft_process = False
    if not skip_rawfft_process:
        print(" [ Server ]: Process FFTs sweeps received from api")
        path_to_api_response = DEF_BASE_PATH + '/resources/api/raw_json/{sname}_{month}_{day}/' \
            .format(sname=inferObj.snsname, month=inferObj.month, day=inferObj.day)
        sps = SpectrumSelector()
        data = sps.build_spc_portion([path_to_api_response], inferObj.freq_start, inferObj.freq_end, n_response_files)
        try:
            os.makedirs(DEF_BASE_PATH + '/resources/api/spectrum_portion/%s/%s_%s/' % (
            inferObj.snsname, inferObj.month, inferObj.day))
        except Exception as error:
            print(error)
        finally:
            spec_portion_path = os.getcwd() + '/Deployment/FlaskApp/resources/api/spectrum_portion/' \
                                              '{snsname}/' \
                                              '{month}_{day}/' \
                                              'SpectrumBands_{startf}_{endf}_{technology}_' \
                                              '{nation}_{startf}_{endf}.npy'.format(snsname=inferObj.snsname,
                                                                                    month=inferObj.month,
                                                                                    day=inferObj.day,
                                                                                    startf=inferObj.startf,
                                                                                    endf=inferObj.endf,
                                                                                    technology=inferObj.technology,
                                                                                    nation=inferObj.nation)
            np.save(spec_portion_path, data)

def serviceTxsDetection():
    print("[ Server ]: Start detection of the transmissions..")
    start = time.time()
    detector.spectrum_path = os.getcwd() + '/Deployment/FlaskApp/resources/api/spectrum_portion/' \
                                           '{snsname}/' \
                                           '{month}_{day}/' \
                                           'SpectrumBands_{startf}_{endf}_{technology}_' \
                                           '{nation}_{startf}_{endf}.npy'.format(snsname=inferObj.snsname,
                                                                                 month=inferObj.month,
                                                                                 day=inferObj.day,
                                                                                 startf=inferObj.startf,
                                                                                 endf=inferObj.endf,
                                                                                 technology=inferObj.technology,
                                                                                 nation=inferObj.nation)

    detector.data = np.load(detector.spectrum_path, allow_pickle=True)[:100, :]
    noise_abs, noise_db = define_noise_level(os.getcwd() + '/Deployment/FlaskApp/resources/api/'
                                                           'raw_json/{snsname}_{month}_{day}/response_1'
                                             .format(snsname=inferObj.snsname, month=inferObj.month, day=inferObj.day))
    detector.tx_detection_funct(noise_db, noiseThres=5, peakThres=3)
    transmissions = detector.channels_detected
    store_transmissions_data(inferObj.det_tx_path, transmissions, detector.data, True, noise_db, inferObj.snsid,
                             inferObj.snsname)
    end = time.time()
    tx_elap_time = np.round(end - start, 3)
    inferObj.txdet_elapsed.append(tx_elap_time)
    print("[ Server ]: End detection of the transmissions..")
    return True

def server_infer_txs():
    start = time.time()
    classifier.spectrum_path = detector.spectrum_path
    classifier.detected_transmissions_path = inferObj.det_tx_path
    classifier.transmissions = detector.channels_detected
    testY_predicted = classifier.inference_trx_labels()
    end = time.time()
    tc_elap_time = np.round(end - start, 3)
    inferObj.tc_elapsed.append(tc_elap_time)
    return testY_predicted

def serviceTechClassif():
    print("[ Server ]: Inference Transmissions")
    pred_labels = server_infer_txs()
    spectrum = detector.data
    print("[ Server ]: Transmissions\n")
    print(classifier.transmissions)
    print("[ Server ]: Inference done. \n")
    # TODO: send json to the frontend
    json = {'pred_labels': pred_labels,
            'Total_res_time': [inferObj.tc_elapsed[0] + inferObj.txdet_elapsed[0]],
            'TC_res_time': inferObj.tc_elapsed,
            'TX_res_time': inferObj.txdet_elapsed,
            'spec_data': spectrum[:50].tolist(),
            'tx_array_bins': classifier.transmissions.tolist(),
            'spec_span_bins': [spectrum.shape[1]]
            }

@app.route('/')
def home():
    return 'App is running'

# Here there are some examples of usage of the API running in the container
# curl -i -H "Content-Type: application/json; charset=utf-8"
# -X POST -data @/Utils/fram_req_template.json http://localhost:5005/services/api/v1.0/usertc/pipeline
# curl -i -H "Content-Type: application/json; charset=utf-8" -X POST -d '{"snsid" :
# "202481596708292", "snsname" : "rack_3", "month" : "May", "day" : "1" , "nation" : "Esp",
# "technology" : "test", "startf" : "20", "endf" : "1500", "freq_start":"20000000",
# "freq_end":"1500000000" }' http://localhost:5005/services/api/v1.0/usertc/pipeline
@app.route(DEF_API_EXPERT_USER + '/pipeline', methods=['POST'])
def service_spectrum_classification():
    global inferObj, classifier, detector
    if "Deployment" in os.getcwd():
        print("[Server]: Change dir")
        time.sleep(2)
        os.chdir('../')
    if not request.json or not 'snsid' in request.json:
        abort(400)

    # 1 - Get sensor and metadata information
    print('[SERVER]: Prediction from teh sensor data:\n')
    inferObj = PredictionInformation()
    datajson = request.json
    inferObj.initialize(datajson)
    print(inferObj.freq_start, inferObj.freq_end)

    # 2 - Creation objects detector and classifiers
    detector = ChannelDetector(5, 1, True, 0.2, True)
    classifier = TechnologyClassifClass(inferObj.freq_start, inferObj.snsid, inferObj.snsname)
    print("[ Server ]: Objects Created: Detector and Classifier")

    # 3.0 - Extraction layer of transmissions
    raw_json_files = 50
    start = time.time()
    print("[ Server ]: Service 1 - Process the FFT chunks")
    serviceProcessRawFFTData(raw_json_files)
    end = time.time()
    rawfft_elapsed_time = np.round(end - start, 3)

    # 3.1 - Transmission Detection
    print("[ Server ]: Service 2 - Transmission Detection")
    serviceTxsDetection()

    # 3.2 - Transmission Classification
    print("[ Server ]: Service 3 Technology Classification ")
    serviceTechClassif()

    # 4 - Print Elapsed time and store the time result
    print("[ Server ] :  FFT, TD, TC Elapsed\n")
    response_time = [rawfft_elapsed_time, inferObj.txdet_elapsed[0], inferObj.tc_elapsed[0]]
    with open('./Deployment/elapsed_time_%d_%s.pkl' % (raw_json_files, inferObj.snsname), 'wb') as fp:
        pickle.dump(response_time, fp)
        print('[ Server ] : Done writing list into a binary file\n')

    # 5 - Return the code 200 for successful service completed
    return '[ Server ]: Service Completed', 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5005, debug=True)  # Run the app in the container

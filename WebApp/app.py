from flask import Flask, render_template, request, jsonify, send_file
import argparse, time, random, string, os, sys, threading, zipfile
import numpy as np
import scipy.signal as scySignal
from rtlsdr import RtlSdr
from datetime import datetime
from math import floor
from utilities import prepare_args, check

# from tensorflow import keras
import tflite_runtime.interpreter as tflite
import subprocess
from BPSK_message_transmission import start_transmitting_bpsk, stop_transmitting_bpsk
import subprocess
import pickle
import dataset2
import processing
import json
import signal

# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.metrics import Accuracy, Precision, Recall

app = Flask(__name__)


# ---------------------- All Routes ------------------


# Home Page
@app.route("/")
def experiments():
    return render_template("experimentMenu.html")


# First Experiment - Signal Classification
@app.route("/signalClassification")
def signalClassification():
    return render_template("signalClassification.html")


@app.route("/test_and_train")
def test_and_train():
    return render_template("trainAndTest.html")


@app.route("/predict_page")
def predict():
    return render_template("predict.html")


@app.route("/signalClassification_train")
def train():
    return render_template("signalClassification_train.html")


# Second Experiment - BPSK Message Transmission
@app.route("/bpsk_message_reception")
def bpsk_message_reception():
    return render_template("bpsk_message_reception.html")


# Third Experiment - BPSK Message Reception
modulations = ["BPSK", "QPSK", "16QAM", "64QAM", "GMSK", "CPFSK", "FSK"]


@app.route("/bpsk_message_transmission")
def bpsk_message_transmission():
    return render_template(
        "bpsk_message_transmission.html", modulations=modulations, message=None
    )


# 5G Signal generation
@app.route("/signal_generator_5g")
def signal_generator():
    return render_template("FiveGSignalGeneration.html")


# Jammer
@app.route("/jammer")
def jammer():
    return render_template("jammer.html")


# Frequency Modulation - Transmission
@app.route("/fmtx")
def fmtx():
    return render_template("fmtx.html")


#  ---------------------------------- All APIs ---------------------------------


# Signal Classification Experiment
# Data Generation
capturing = False
capture_thread = None
@app.route("/predict", methods=["POST"])
def predict_model():
    args = prepare_args()
    data = request.json
    model_name = data.get("model", "highSNR_Model")
    modelType = data.get("modelType")
    freq = float(data["freq"])
    sample_rate = int(data["sample_rate"])
    decimation_rate = int(data["decimation_rate"])
    sdr = 1  # int(data['sdr'])
    classes = [
        "bpsk",
        "qpsk",
        "wfm",
        "sdfg",
        "sgs",
        "sgsfd",
        "sgf",
    ]  #  [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))] # Gets classes from all the folders mentioned under training_data folder

    model_path = "models/" + model_name  # Folder for the data used for training
    # model = keras.models.load_model(model_path + model_name)

    freq_mhz, maxlabel, maxim = 0, 0, 0
    if modelType == "tflite":
        freq_mhz, maxlabel, maxim = tflite(
            freq, sample_rate, decimation_rate, sdr, classes, model_path
        )
    elif modelType == "knn":
        freq_mhz, maxlabel, maxim = tflite(
            freq, sample_rate, decimation_rate, sdr, classes, model_path
        )
    else:
        pass

    return jsonify({"frequency_mhz": freq_mhz, "label": maxlabel, "confidence": maxim})


@app.route("/start", methods=["POST"])
def start_capture():
    global capturing, capture_thread
    if capturing:
        return jsonify({"status": "Capturing already in progress"}), 400

    data = request.json
    classname = data["classname"]
    freq = float(data["freq"])  # get_frequency(classname)
    num_classes = 1  # int(data['num_classes'])
    err_ppm = 56  # int(data['err_ppm'])
    numberOfSamples = int(data["samples"])
    gain = int(data["gain"])
    numberOfsamplesForTrainingAndTesting = int(data["samplesForTrainingAndTesting"])
    percentageForTesting = int(data["percentageForTestData"])
    decimation_rate = int(data["decimation_rate"])
    sample_rate = int(data["sample_rate"])
    sdr = 1  # int(data['sdr'])

    capturing = True
    # capture_thread = threading.Thread(target=collect_samples, args=(freq, gain, classname, decimation_rate, sample_rate, err_ppm, sdr, num_classes, numberOfSamples, numberOfsamplesForTrainingAndTesting, percentageForTesting))
    # capture_thread.start()

    collect_samples(
        freq,
        gain,
        classname,
        decimation_rate,
        sample_rate,
        err_ppm,
        sdr,
        num_classes,
        numberOfSamples,
        numberOfsamplesForTrainingAndTesting,
        percentageForTesting,
    )
    return jsonify({"status": "Capture started"})


@app.route("/stop", methods=["POST"])
def stop_capture():
    global capturing, capture_thread
    capturing = False
    if capture_thread:
        capture_thread.join()
    return jsonify({"status": "Capture stopped"})


@app.route("/download", methods=["POST"])
def download_data():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"data_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for folder in ["training_data", "testing_data"]:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    zipf.write(os.path.join(root, file))
    return send_file(zip_filename, as_attachment=True)


# Training
@app.route("/start_training", methods=["POST"])
def start_training():
    try:
        classes = request.form["classes"].split(",")
        validation_size = float(request.form["validation_size"])
        DIM1 = int(request.form["DIM1"])
        DIM2 = int(request.form["DIM2"])
        optimizer = request.form["optimizer"]
        metrics = request.form["metrics"]
        epochs = int(request.form["epochs"])
        batch_size = int(request.form["batch_size"])
        shuffle = "shuffle" in request.form
        train_path = request.form["train_path"]

        train_path = train_path if train_path else "training_data"
        num_classes = len(classes)
        data = dataset2.read_train_sets2(
            train_path, classes, validation_size=validation_size
        )

        Xtrain = data.train.images
        Ytrain = data.train.labels
        Xtest = data.valid.images
        Ytest = data.valid.labels

        input_signal = Input(shape=(DIM1, DIM2, 2))
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(input_signal)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=input_signal, outputs=x)

        if optimizer == "adam":
            opt = Adam()
        elif optimizer == "sgd":
            opt = SGD()
        elif optimizer == "rmsprop":
            opt = RMSprop()

        if metrics == "accuracy":
            metric = Accuracy()
        elif metrics == "precision":
            metric = Precision()
        elif metrics == "recall":
            metric = Recall()

        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[metric])
        history = model.fit(
            Xtrain,
            Ytrain,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=(Xtest, Ytest),
        )

        model_file = "model.h5"
        model.save(model_file)
        pickle.dump(history.history, open("history.pkl", "wb"))

        return jsonify(
            {
                "message": "Training completed successfully.",
                "model_generated": True,
                "model_file": model_file,
            }
        )
    except Exception as e:
        return jsonify(
            {
                "message": f"An error occurred during training: {str(e)}",
                "model_generated": False,
            }
        )


@app.route("/download_model", methods=["GET"])
def download_model():
    model_file = request.args.get("file")
    if model_file and os.path.exists(model_file):
        return send_file(model_file, as_attachment=True)
    return "File not found", 404


# Predict
@app.route("/upload_model", methods=["POST"])
def upload_model():
    if "model-file" not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files["model-file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    if file:
        filename = file.filename
        filepath = os.path.join("models", filename)
        file.save(filepath)
        return (
            jsonify({"message": "Model uploaded successfully", "filename": filename}),
            200,
        )


# BPSK Message reception
bpsk_reception_process = None
decoded_message = ""


@app.route("/bpsk_start_receiving", methods=["POST"])
def bpsk_start_message_reception():
    global bpsk_reception_process
    global decoded_message

    data = request.json
    samp_rate = data["samp_rate"]
    packet_len = data["packet_len"]
    center_freq = data["center_freq"]
    gain = data["gain"]

    # Start the decoding process
    bpsk_reception_process = subprocess.Popen(
        [
            "python3",
            "BPSK_message_decoder_no_gui.py",
            samp_rate,
            packet_len,
            center_freq,
            gain,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    decoded_message = ""

    return jsonify({"decoded_message": decoded_message})


@app.route("/bpsk_stop_receiving", methods=["POST"])
def bpsk_stop_message_reception():
    global bpsk_reception_process
    global decoded_message

    if bpsk_reception_process:
        bpsk_reception_process.send_signal(signal.SIGINT)
        stdout, stderr = bpsk_reception_process.communicate()
        decoded_message = stdout.decode("utf-8")
        bpsk_reception_process = None

    return jsonify({"decoded_message": decoded_message})


# BPSK Message transmission
@app.route("/start", methods=["POST"])
def bpsk_start_message_transmission():
    char_to_transmit = request.form["char_to_transmit"]
    modulation = request.form["modulation"]
    gain = float(request.form["gain"])
    symbol_period = float(request.form["symbol_period"])

    start_transmitting_bpsk(char_to_transmit, modulation, gain, symbol_period)
    return jsonify({"status": "transmitting"})


@app.route("/stop", methods=["POST"])
def bpsk_stop_message_transmission():
    stop_transmitting_bpsk()
    return jsonify({"status": "stopped"})


# 5G Signal Generation
@app.route("/generate_5g_signal", methods=["POST"])
def generate_signal():
    try:
        mu = int(request.form["mu"])
        nrb = int(request.form["nrb"])
        guard = int(request.form["guard"])
        total_length = int(request.form["total_length"])
        nfft = int(request.form["nfft"])
        sequence_str = request.form["sequence"]
        cpsize = 144 #int(request.form["cpsize"])
        numFrames = int(request.form["numFrames"])

        sequence = eval(sequence_str)  # Use eval to convert string to list
        filename = processing.generate_signal(
            nfft, 1, mu, nrb, guard, 0, total_length, sequence, cpsize, numFrames
        )

        return jsonify(
            {"message": f"Success: File saved as {filename}", "filename": f"{filename}"}
        )
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


@app.route("/open_in_inspectrum", methods=["POST"])
def open_inspectrum():
    try:
        data = request.json
        filename = data["filename"]
        if filename:
            subprocess.run(["E:/ProgramFiles/PothosSDR/bin/inspectrum.exe", filename])
            return jsonify({"message": "Inspectrum opened successfully."})
        else:
            return jsonify({"message": "No filename provided."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


@app.route("/download_iq_file", methods=["GET"])
def download_file():
    filename = request.args.get("filename")
    print("Filename : ", filename)
    if filename and os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "File not found", 404


# Jammer
jammer_process = None


@app.route("/start_jammer", methods=["POST"])
def start_jammer():
    try:
        global jammer_process

        sample_rate = request.form["sample_rate"]
        tx_gain = request.form["tx_gain"]
        num_samples = request.form["num_samples"]
        hop_interval = request.form["hop_interval"]
        freq_list = request.form["freq_list"].split(",")

        # Start the SDR process
        jammer_process = subprocess.Popen(
            [
                "python3",
                "jammer.py",
                sample_rate,
                tx_gain,
                num_samples,
                hop_interval,
                json.dumps(freq_list),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return jsonify(
            {
                "message": "SDR process started successfully.",
                "process_id": jammer_process.pid,
            }
        )
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


@app.route("/stop_jammer", methods=["POST"])
def stop_jammer():
    try:
        global jammer_process

        if jammer_process:
            jammer_process.send_signal(signal.SIGINT)
            jammer_process = None
            return jsonify({"message": "SDR process stopped successfully."})
        else:
            return jsonify({"message": "No SDR process is running."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


# Frequency Modulation - Transmission
fmtx_process = None


@app.route("/start_fmtx", methods=["POST"])
def start_fmtx():
    try:
        global fmtx_process

        samp_rate = request.form["samp_rate"]
        audio_rate = request.form["audio_rate"]
        center_freq = request.form["center_freq"]
        gain = request.form["gain"]
        bb_gain = request.form["bb_gain"]
        if_gain = request.form["if_gain"]
        bandwidth = request.form["bandwidth"]
        wavfile = request.form["wavfile"]

        # Start the FM transmission process
        fmtx_process = subprocess.Popen(
            [
                "python3",
                "fmtx_script.py",
                samp_rate,
                audio_rate,
                center_freq,
                gain,
                bb_gain,
                if_gain,
                bandwidth,
                wavfile,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return jsonify(
            {
                "message": "FM transmission started successfully.",
                "process_id": fmtx_process.pid,
            }
        )
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


@app.route("/stop_fmtx", methods=["POST"])
def stop_fmtx():
    try:
        global fmtx_process

        if fmtx_process:
            fmtx_process.send_signal(signal.SIGINT)
            fmtx_process = None
            return jsonify({"message": "FM transmission stopped successfully."})
        else:
            return jsonify({"message": "No FM transmission is running."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"})


# ------------------------------ All Utitlity functions ----------------------


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def collect_samples(
    freq,
    gain,
    classname,
    decimation_rate,
    sample_rate,
    err_ppm,
    sdr,
    num_classes,
    numberOfSamples,
    numberOfsamplesForTrainingAndTesting,
    percentageForTesting,
):
    global capturing
    os.makedirs("training_data/" + classname, exist_ok=True)
    os.makedirs("testing_data/" + classname, exist_ok=True)
    for i in range(0, numberOfsamplesForTrainingAndTesting):
        if not capturing:
            break
        if sdr == 1:
            iq_samples = read_samples_sdr(
                freq, gain, sample_rate, err_ppm, numberOfSamples
            )
        elif sdr == 0:
            # iq_samples = read_samples(freq, sample_rate)
            pass
        iq_samples = scySignal.decimate(iq_samples, decimation_rate, zero_phase=True)
        if i < floor(
            (percentageForTesting / 100) * numberOfsamplesForTrainingAndTesting
        ):  # 75% train, 25% test
            filename = (
                "training_data/" + classname + "/samples-" + randomword(16) + ".npy"
            )
        else:
            filename = (
                "testing_data/" + classname + "/samples-" + randomword(16) + ".npy"
            )
        np.save(filename, iq_samples)
        if not (i % 5):
            print(i / 5, "%", classname)
    capturing = False
    # return jsonify({'status': 'Capture started'})


def read_samples_sdr(freq, gain, sample_rate, err_ppm, numberOfSamples):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.err_ppm = err_ppm
    sdr.gain = gain if gain is not None else "auto"

    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(
        -1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples))
    )  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

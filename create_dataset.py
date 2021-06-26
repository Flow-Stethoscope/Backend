import json

import librosa
import librosa.display

from scipy.signal import butter, lfilter
import math

from pathlib import Path
import csv

DOWN_SR = 1000
BPF_LOW = 20
BPF_HIGH = 400
BPF_ORDER = 3

AUDIO_LEN = 5  # in seconds


def create_dataset(dataset_dir, output_path):
    data = {"mfccs": [], "labels": []}

    for dataset_path in dataset_dir.iterdir():
        if "training-" in dataset_path.name:
            print(f"Processing training set at {dataset_path}.")
            with open(dataset_path / "REFERENCE.csv", "r") as f:
                csv_reader = csv.reader(f)
                for file_name, label in csv_reader:
                    file_path = str(dataset_path / file_name) + ".wav"

                    # print(f"Processing {file_name} at location {file_path}.")

                    mfccs = process_audio_file(file_path, AUDIO_LEN)
                    data["mfccs"] += mfccs
                    data["labels"] += [int(label)] * len(mfccs)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def process_audio_file(audio_path, sample_len):
    audio, _ = librosa.load(audio_path, sr=DOWN_SR)  # load + down sample
    return process_audio_np(audio, sample_len)


def process_audio_np(audio_np, sample_len):
    audio = denoise_audio_np(audio_np)

    n_fft = int(0.025 * DOWN_SR)
    hop_len = int(0.01 * DOWN_SR)

    num_segments = int((audio.size / DOWN_SR) // sample_len)
    samples_per_segment = sample_len * DOWN_SR
    expected_mfcc_per_segment = math.ceil(samples_per_segment / hop_len)

    mfccs = []
    for current_segment in range(num_segments):
        start = samples_per_segment * current_segment
        end = start + samples_per_segment

        # find mfcc
        mfcc = librosa.feature.mfcc(
            audio[start:end], DOWN_SR, n_fft=n_fft, hop_length=hop_len, n_mfcc=13
        )
        mfcc = mfcc.T

        if len(mfcc) == expected_mfcc_per_segment:
            mfccs.append(mfcc.tolist())

    return mfccs


def denoise_audio_np(audio_np, bpf_low=20, bpf_high=400, bpf_order=3):
    audio = butter_bandpass_filter(audio_np, bpf_low, bpf_high, DOWN_SR, bpf_order)
    return 2 * (audio - audio.min()) / (audio.max() - audio.min()) - 1.0  # normalize


# bandpass from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    datasets_path = Path("./datasets/classification-heart-sounds-physionet/")
    data_output_path = Path("./datasets/data.json")

    create_dataset(datasets_path, data_output_path)

    # to unjsonify numpy arrays: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

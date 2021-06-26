from pathlib import Path

import soundfile
import tensorflow.keras as keras
import numpy as np
import librosa

from create_dataset import process_audio_np, denoise_audio_np


class Inference:
    def __init__(self, model_path=Path("./HeartSounds_BinaryClassification_BLSTM"), sample_rate=1000):
        self.model = keras.models.load_model(model_path)
        self.sample_rate = sample_rate

        self.model.summary()

    def predict(self, byte_list, threshold=0.05):
        if len(byte_list) / self.sample_rate < 5:
            return "unsure"

        x = self.preprocess(byte_list)

        pred = self.model.predict_on_batch(x)
        avg_pred = np.mean(pred)

        if 0.5 - threshold <= avg_pred <= 0.5 + threshold:
            return "unsure"
        if avg_pred > 0.5:
            return "abnormal"
        return "normal"

    def preprocess(self, byte_list):
        audio = np.array(byte_list)
        audio = librosa.resample(audio, self.sample_rate, 1000)
        mfccs = process_audio_np(audio, 5)  # 5 second samples
        return np.array(mfccs)

    def get_wav(self, byte_list, wav_path=Path("./tmp/tmp_recording.wav")):
        audio = np.array(byte_list)
        audio = librosa.resample(audio, self.sample_rate, 1000)
        soundfile.write(wav_path, audio, 1000)
        return wav_path

    def delete_wav(self, wav_path=Path("./tmp/tmp_recording.wav")):
        wav_path = Path(wav_path)
        wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    model_full_save = Path(
        "./model_saves/3523832_epochs_15-batch_size_2-lr_1e-06/full_save"
    )
    test_audio_path = Path(
        "./datasets/classification-heart-sounds-physionet/training-a/a0004.wav"
    )

    inference = Inference(
        model_full_save, 2000
    )  # for testing, dataset has 2kHz sample rate

    audio_np, sr = librosa.load(test_audio_path, sr=None)
    wav = inference.get_wav(audio_np.tolist())
    print(wav)

    inference.delete_wav(wav)

    print(inference.predict(audio_np))

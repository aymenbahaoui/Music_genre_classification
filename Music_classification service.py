import librosa
import tensorflow as tf
import numpy as np
from youtube_to_wav import *
SAMPLES_TO_CONSIDER = 22050 * 3
SAVED_MODEL_PATH = "model.h5"

class _Music_Classification_Service:
    model = None
    _mapping = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"

    ]
    _instance = None

    def predict(self, file_path):
        # extract MFCC
        MFCCs = self.preprocess(file_path)
        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        #predicted_index = np.argmax(predictions)
        #predicted_genre = self._mapping[predicted_index]
        #return predicted_genre
        return predictions[0]

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[SAMPLES_TO_CONSIDER*10:SAMPLES_TO_CONSIDER*11]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Music_Classification_Service():
    # ensure an instance is created only the first time the factory function is called
    if _Music_Classification_Service._instance is None:
        _Music_Classification_Service._instance = _Music_Classification_Service()
        _Music_Classification_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Music_Classification_Service._instance


if __name__ == "__main__":
    # create 2 instances of the keyword spotting service
    mcs = Music_Classification_Service()
    mcs1 = Music_Classification_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert mcs is mcs1

    # make a prediction
    vidName = youtube_to_wav("https://www.youtube.com/watch?v=fh-o8Bxc3Ys")
    genre = mcs.predict(vidName)
    for i in range(len(genre)):
        print(str(mcs._mapping[i]) + "--->" + str(genre[i]))
    os.remove(vidName)
import json
import os
import math
import librosa

DATASET_PATH = "D:\\soud of IA\\genres\\genres"
JSON_PATH = "data_10.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extrait MFCCs de la musique du dataset et les enregister sous forme d'un fichier avec leurs labels.
        :param dataset_path (str): le chemin vers notre dataset
        :param json_path (str): le chemin vers notre fichier json
        :param num_mfcc (int): nombre de coefficient MFCC à extraire
        :param n_fft (int): Intervalle à consideimr pour appliquer le FFT. Mesuré en # d'échantillons
        :param hop_length (int): Sliding window de FFT. Mesuré en # d'échantillons
        :param: num_segments (int): Nombre de segments dans lesquels nous voulons diviser les pistes
        :return:
        """

    # dictionary pour enregistrer mapping, labels, et MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # boucler suivant tous les sous dossier des genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensurer on parcours un dossier du genre
        if dirpath is not dataset_path:

            # enregistrer l'étiquette de genre (c'est-à-dire le nom du sous-dossier) dans le mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # traiter tous les fichier dans genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):

                    # calculer le debut et la fin de chaque segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extraire mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # enregistrer le MFCC
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # enregistrer le data en json
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
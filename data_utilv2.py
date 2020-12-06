import tensorflow_datasets as tfds
import tensorflow as tf
import json
from glob import glob
from pathlib import Path
from scipy.io import wavfile
import random
import os
import shutil

sample_rate = 16000


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def train_test_val_split(path_to_wavs):
    if path_to_wavs[-1] != '/':
        path_to_wavs = path_to_wavs + '/'

    # Split to train, test, val
    file_paths = set(glob(path_to_wavs + "*.wav"))
    total_len = len(file_paths)

    # 80% train
    train_paths = set(random.sample(file_paths, int(0.8 * total_len)))

    # 20% test -> 10% test + 10% val
    test_paths = file_paths - train_paths
    val_paths = set(random.sample(test_paths, int(0.5 * len(test_paths))))
    test_paths = test_paths - val_paths

    return train_paths, test_paths, val_paths


def create_ds_folder(train, test, val):
    try:
        shutil.rmtree('reverbsynth_raw')
    except:
        pass
    try:
        os.makedirs('reverbsynth_raw/train')
        os.makedirs('reverbsynth_raw/test')
        os.makedirs('reverbsynth_raw/valid')
    except:
        pass


def to_tfrecord(audio, pitch):
    feature = {
        'audio': _float_feature(audio),  # audio is a list of floats
        'pitch': _int64_feature(pitch)  # wrap pitch index in list
    }
    # Example is a flexible message type that contains key-value pairs,
    # where each key maps to a Feature message. Here, each Example contains
    # two features: A FloatList for the decoded audio data and an Int64List
    # containing the corresponding label's index.
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_records(paths, metadata, output_path):
    global sample_rate
    with tf.io.TFRecordWriter(output_path) as writer:
        # Write records
        for index, path in enumerate(paths):
            # read audio file
            sample_rate, audio_array = wavfile.read(path)

            # select only the first 64000 entries
            audio_array = audio_array[:64000]

            key = Path(path).stem
            data = metadata[key]
            pitch = data['pitch']

            example = to_tfrecord(audio_array, pitch)
            writer.write(example.SerializeToString())


def generate_tfrecords(path_to_wavs, path_to_meta):
    train_paths, test_paths, val_paths = train_test_val_split(path_to_wavs)
    print(f"train_records: {len(train_paths)}\ntest_records: {len(test_paths)}\nval_records: {len(val_paths)}")
    # create raw dataset
    create_ds_folder(train_paths, test_paths, val_paths)

    # Load metadata
    with open(path_to_meta) as f:
        metadata = json.load(f)

    print("Generating train records")
    write_records(train_paths, metadata, 'reverbsynth_raw/train/reverbsynth-train.tfrecord')
    print("Generating test records")
    write_records(test_paths, metadata, 'reverbsynth_raw/test/reverbsynth-test.tfrecord')
    print("Generating val records")
    write_records(val_paths, metadata, 'reverbsynth_raw/valid/reverbsynth-valid.tfrecord')


generate_tfrecords(r'C:\Users\ghana\Documents\DL\new_data\audio', r'C:\Users\ghana\Documents\DL\new_data\examples.json')

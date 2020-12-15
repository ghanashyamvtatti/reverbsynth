# ReverbSynth: Applying GANSynth to Audio DSP

## Requirements
* Python3.5+
* ffmpeg
* rtmidi
* [SuperCollider](https://supercollider.github.io/download)

## Instructions
### Setup
* Install SuperCollider, python and ffmpeg
* Clone this repository
* CD into this repository
* Run `pip install -r requirements.txt`
* Clone [this](https://github.com/ghanashyamvtatti/magenta) repository
* CD into it and run `pip install -e .`

Note: You **MUST** use the magenta from the above repo and not the original one.

### Creating audio with reverb using SuperCollider
* Open the batchNRT.scd file in SuperCollider
* Change the input and output paths as needed
* Execute the first code block, which contains the reverb synthdef (you can substitute other synthdefs in here, too)
* Execute the second code block, which contains the actual batch processing script
* Move the clean_wavs.sh script into the output folder where all the .wav files are and run it.

Credit where credit is due: This script is an update and repurposing of [Dan Stowell's batchNRT quark](https://github.com/supercollider-quarks/batchNRT)

### Converting wav files into tfrecords
* Construct the metadata (just the pitch is sufficient) and put it in a json as done in [NSynth](https://magenta.tensorflow.org/datasets/nsynth#example-features)
* Run `python generate_tfrecords.py` and enter the paths.

### Training
* CD into the cloned magenta repo's `magenta/magenta/models/gansynth` path
* Run `python gansynth_train.py --hparams='{"tfds_data_dir":"<path to train tfrecord>", "dataset_name":"reverbsynth", "train_root_dir":"<path to save outputs>"}' --config=mel_prog_hires`
* The training takes around 3-4 days on a V100 GPU.

### Stats and Outputs
* You can track the losses and the generated audio quality as the training progresses using tensorboard.
* Run `tensorboard --logdir <path to train outputs directory>`
* You'll find the audio samples in the Audio tab.

### Generation
* CD into the cloned magenta folder.
* Run `python magenta/models/gansynth/gansynth_generate.py --ckpt_dir=<training output directory> --output_dir=<output directory> --midi_file=<path to input midi file>`

## Examples
You'll find the sample generated outputs in the examples folder.

## Pretrained
The trained bass reverbsynth model and training outputs is available [here](https://drive.google.com/file/d/17xPan4KpZ1OkWnY1_VZQcWsoyUHWBhL5/view?usp=sharing).

It can be used directly for generation and with tensorboard.

## References
* GANSynth: Adversarial Neural Audio Synthesis 
Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, Adam Roberts

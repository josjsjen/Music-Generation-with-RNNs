# Music Generation with RNNs
Implementation of Recurrent Neural Network (RNN) for music generation.

We will be using the MIDI music toolkit. Please run the following cell to confirm that you have the midi package, which allows us to use the MIDI music tools in Python.

```bash
pip install python-midi
```

## Data Cleaning
You can find in create_dataset.py and midi_manipulation.py for data cleaning.

## Dataset
The dataset for this lab will be taken from the data/ folder. The dataset we have downloaded is a set of pop song snippets. If you double-click any of the MIDI files, you can open them with a music playing app such as GarageBand (Mac) or MuseScore.

The dataset is a list of np.arrays, one for each song. Each song should have the shape (song_length, num_possible_notes), where song_length >= min_song_length. The individual feature vectors of the notes in the song are processed into a one-hot encoding, meaning that they are binary vectors where one and only one entry is 1.

## Model
You can find the details in mu

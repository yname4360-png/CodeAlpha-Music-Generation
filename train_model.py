import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from midi_utils import extract_notes

SEQUENCE_LENGTH = 40
EPOCHS = 50
BATCH_SIZE = 64

print("Reading MIDI files...")
notes = extract_notes()

pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - SEQUENCE_LENGTH):
    sequence = notes[i:i + SEQUENCE_LENGTH]
    target = notes[i + SEQUENCE_LENGTH]

    network_input.append([note_to_int[x] for x in sequence])
    network_output.append(note_to_int[target])

n_patterns = len(network_input)

if n_patterns < 100:
    raise ValueError(
        f"Only {n_patterns} training sequences were created. "
        "Add more MIDI files to dataset/midi_files/."
    )

X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
X = X / float(len(pitchnames))
y = to_categorical(network_output, num_classes=len(pitchnames))

print(f"Training sequences: {n_patterns}")
print(f"Unique notes/chords: {len(pitchnames)}")

model = Sequential([
    LSTM(256, input_shape=(SEQUENCE_LENGTH, 1), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(len(pitchnames), activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

os.makedirs("model", exist_ok=True)

model.fit(
    X,
    y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

model.save("model/music_model.keras")

with open("model/notes.pkl", "wb") as f:
    pickle.dump(pitchnames, f)

print("\nTraining complete!")
print("Model saved: model/music_model.keras")
print("Notes saved: model/notes.pkl")

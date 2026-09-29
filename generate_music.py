import pickle
import numpy as np
from music21 import note, chord, stream
from tensorflow.keras.models import load_model
from midi_utils import extract_notes

SEQUENCE_LENGTH = 40
GENERATE_NOTES = 200

MODEL_PATH = "model/music_model.keras"
NOTES_PATH = "model/notes.pkl"

print("Loading model...")
model = load_model(MODEL_PATH)

with open(NOTES_PATH, "rb") as f:
    pitchnames = pickle.load(f)

note_to_int = {name: i for i, name in enumerate(pitchnames)}

# Use the beginning of the available MIDI data as a seed.
all_notes = extract_notes()
seed = all_notes[:SEQUENCE_LENGTH]

if len(seed) < SEQUENCE_LENGTH:
    raise ValueError("Not enough notes available for generation.")

pattern = [note_to_int.get(x, 0) for x in seed]
prediction_output = []

for _ in range(GENERATE_NOTES):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(prediction_input, verbose=0)
    index = int(np.argmax(prediction))
    result = pitchnames[index]

    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

output_notes = []

for pattern_item in prediction_output:
    if "." in pattern_item:
        notes_in_chord = pattern_item.split(".")
        chord_notes = [
            note.Note(int(n)) for n in notes_in_chord
        ]
        new_chord = chord.Chord(chord_notes)
        output_notes.append(new_chord)
    else:
        output_notes.append(note.Note(pattern_item))

midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="generated_music.mid")

print("\nMusic generated successfully!")
print("Output file: generated_music.mid")

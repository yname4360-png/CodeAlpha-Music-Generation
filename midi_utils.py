from music21 import converter, instrument, note, chord
import glob

def extract_notes(midi_folder="dataset/midi_files"):
    notes = []
    files = glob.glob(f"{midi_folder}/**/*.mid", recursive=True)
    files += glob.glob(f"{midi_folder}/**/*.midi", recursive=True)

    if not files:
        raise FileNotFoundError(
            "No MIDI files found. Put .mid/.midi files inside dataset/midi_files/"
        )

    for file in files:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flatten().notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
        except Exception as exc:
            print(f"Skipping {file}: {exc}")

    if not notes:
        raise ValueError("MIDI files were found, but no usable notes were extracted.")

    return notes

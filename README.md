# AI Music Generation using LSTM

This project generates new musical sequences using an LSTM-based deep learning model.

## Internship Task
CodeAlpha - Artificial Intelligence Internship  
Task 3: Music Generation with AI

## Technologies
- Python
- TensorFlow / Keras
- LSTM
- music21
- NumPy
- MIDI

## Project Flow

MIDI Dataset
   ↓
Extract Notes
   ↓
Create Training Sequences
   ↓
LSTM Neural Network
   ↓
Train Model
   ↓
Generate New Notes
   ↓
Create MIDI File

## Folder Structure

```text
CodeAlpha_MusicGeneration/
├── dataset/
│   └── midi_files/
├── model/
├── midi_utils.py
├── train_model.py
├── generate_music.py
├── requirements.txt
└── README.md
```

## Setup

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Put multiple `.mid` or `.midi` files inside:

```text
dataset/midi_files/
```

Use royalty-free or legally usable MIDI files for your project.

## Train

```bash
python train_model.py
```

After training, these files will be created:

```text
model/music_model.keras
model/notes.pkl
```

## Generate Music

```bash
python generate_music.py
```

The generated file will be:

```text
generated_music.mid
```

## Notes

- More MIDI files generally provide more training patterns.
- Training time depends on your computer and dataset size.
- This is an educational project demonstrating sequence generation with LSTM.


import os, math, string, pefile, time, threading
import tkinter as tk
import numpy as np
from capstone import *
from flask import Flask, render_template, request, jsonify
import subprocess
# from keras.models import Sequential, Model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input


from keras import layers, preprocessing
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames
from tkinter.ttk import Progressbar

import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from tensorflow.keras.layers import Input
## Defining models (opcode, strings, and ensemble)

# Defining the opcode model
opModel = Sequential()

opModel.add(layers.InputLayer(input_shape=(50,)))
opModel.add(layers.Dense(256, activation='relu'))
opModel.add(layers.BatchNormalization())
opModel.add(layers.Dense(128, activation='relu'))
opModel.add(layers.BatchNormalization())
opModel.add(layers.Dense(64, activation='relu'))
opModel.add(layers.BatchNormalization())
opModel.add(layers.Dense(32, activation='relu'))
opModel.add(layers.BatchNormalization())
opModel.add(layers.Dense(16, activation='relu'))
opModel.add(layers.BatchNormalization())
opModel.add(layers.Dense(3, activation='softmax'))

opModel.load_weights("weights-improvement-574-0.85.hdf5")

opModel.compile(optimizer="rmsprop",
              loss='categorical_crossentropy',
              metrics=['accuracy'])


class histSequence(Sequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = shuffle(x, y)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            np.load(file_name)
            for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        pass


class histSequenceVal(histSequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size


# Defining the strings as greyscale images model
model = Sequential()

model.add(layers.InputLayer(input_shape=(100, 100, 1)))
model.add(layers.SpatialDropout2D(rate=0.2))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.SpatialDropout2D(rate=0.1))
model.add(layers.Conv2D(16, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.SpatialDropout2D(rate=0.1))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))


class hashCorpusSequence(Sequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = shuffle(x, y)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([
            np.rint(((np.load(file_name) - np.min(np.load(file_name))) /
            (np.max(np.load(file_name)) - np.min(np.load(file_name)))) * 255).astype(int)
            for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        pass


class hashCorpusSequenceVal(hashCorpusSequence):

    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size


model.load_weights("weights-improvement-04-0.72.hdf5")

model.compile(optimizer="adamax",
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# opModel.name = "opcodeModel"
# model.name = "stringsAsGreyscaleModel"

def ensemble(models, model_inputs):
    outputs = [models[0](model_inputs[0]), models[1](model_inputs[1])]
    y = layers.average(outputs)

    modelEns = Model(model_inputs, y, name='ensemble')

    return modelEns


models = [opModel, model]
model_inputs = [Input(shape=(50,)), Input(shape=(100, 100, 1))]
modelEns = ensemble(models, model_inputs)
modelEns.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


## Pre-processing of PE (EXE, DLL, etc.) file(s)

# https://stackoverflow.com/questions/17195924/python-equivalent-of-unix-strings-utility
# Solution to Python based 'strings' alternative from SO. Decodes bytes of binary file as
# utf-8 strings
def strings(filename, min=4):
    with open(filename, errors="ignore", encoding="utf-8") as f:
        result = ""
        for c in f.read():
            if c in string.printable:
                result += c
                continue
            if len(result) >= min:
                yield result
            result = ""
        if len(result) >= min:  # catch result at EOF
            yield result


# Converting utf-8 string to sequence of words
def wordSequence(pePath):
    try:
        text = ""
        for s in strings(pePath):
            text += s + "\n"
        sequence = preprocessing.text.text_to_word_sequence(text)[:10000]
        return sequence
    except Exception as e:
        print(e)

# Hashing words of word sequences into sequences of word-specific integers
def hashWordSequences(sequences, maxSeqLen, vocabSize):

    hashedSeqs = []
    docCount = 0
    for sequence in sequences:
        try:
            text = " ".join(sequence)
            hashWordIDs = preprocessing.text.hashing_trick(text, round(vocabSize * 1.5), hash_function='md5')
            docLen = len(hashWordIDs)
            if docLen < maxSeqLen:
                hashWordIDs += [0 for i in range(0, maxSeqLen-docLen)]
            hashWordIDs = np.array(hashWordIDs).reshape(100, 100, 1)
            hashedSeqs.append(hashWordIDs)
            docCount += 1
        except Exception as e:
            print(e)
    return hashedSeqs


# Function takes list of paths to PE files and returns a list
# of lists, with the first index as input for the opcode model,
# and the second index as input for the strings model
def preprocess_single_pe(pePaths):
    mlInputs = []

    for sample in pePaths:
        try:
            pe = pefile.PE(sample, fast_load=True)
            entryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            data = pe.get_memory_mapped_image()[entryPoint:]
            cs = Cs(CS_ARCH_X86, CS_MODE_32)

            opcodes = [i.mnemonic for i in cs.disasm(data, 0x1000)]
            
            opFreqVec = [opcodes.count(opcode) for opcode in set(opcodes)]
            
            # Pad to ensure length is at least 50
            opFreqVec = opFreqVec + [0] * max(0, 50 - len(opFreqVec))
            
            # Trim to ensure length is exactly 50
            opFreqVec = opFreqVec[:50]
            
            mlInputs.append(opFreqVec)

        except Exception as e:
            print(e)

    mlInputs = np.array(mlInputs)

    # Assuming sequences is a list of word sequences
    sequences = [wordSequence(sample) for sample in pePaths]
    
    # Assuming hashWordSequences returns a list of hashed sequences
    hashSeqs = hashWordSequences(sequences, 10000, 15000)  # Adjust maxVocabSize accordingly

    # Convert mlInputs to a list containing two elements
    mlInputs = [mlInputs, hashSeqs]

    return mlInputs



## Function taking paths to PE files as input, and returning ensemble model predictions
# as output
def predictPEs(pePaths):
    classNames = ["benign", "malware", "ransomware"]
    pePredictions = {}

    count = 0
    for pePath in pePaths:
        x1 = preprocessPEs(pePaths)[count][0].reshape(1, 50)
        x2 = preprocessPEs(pePaths)[count][1].reshape(1, 100, 100, 1)
        count += 1
        pePredictions[pePath] = classNames[np.argmax(modelEns.predict(x=[x1, x2]))]

    return pePredictions


def predict_single_pe(pe_path):
    pePaths = [pe_path]

    try:
        # Preprocess the single PE file
        preprocessed_data = preprocess_single_pe(pePaths)
        x1 = preprocessed_data[0][0].reshape(1, 50)
        x2 = preprocessed_data[1][0].reshape(1, 100, 100, 1)

        # Define class names
        classNames = ["benign", "malware", "ransomware"]

        # Make the prediction using the ensemble model
        prediction = modelEns.predict(x=[x1, x2])

        # Get the predicted class
        predicted_class = classNames[np.argmax(prediction)]

        return {'predicted_class': predicted_class}

    except Exception as e:
        return {'error': str(e)}


# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python ensemblePredict.py <path_to_pe_file>")
#         sys.exit(1)

#     pe_path = sys.argv[1]
#     if not os.path.isfile(pe_path):
#         print(f"Error: File not found: {pe_path}")
#         sys.exit(1)

#     result = predict_single_pe(pe_path)
#     print(result)

if __name__ == "__main__":
    # Add any specific code for standalone execution if needed
    pass
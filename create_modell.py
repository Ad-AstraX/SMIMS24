#!/usr/bin/env python3
from base import NeuralNetwork
from pathlib import Path
from PIL import Image
import numpy as np
import random


def get_targets(length, right_index):
    targets = np.zeros(length) + 0.01
    targets[right_index] = 0.99

    return targets


BASE_DIRECTORY = Path('C:\\Plant_leave_diseases_dataset_with_augmentation')
NUMBER_INODES = 256 * 256
NUMBER_HNODES = 100
LEARNING_RATE = 0.1
cases_parsed = 0
data_pairs = []

for i, path in enumerate(BASE_DIRECTORY.glob('*')):
    if not 'Apple' in path.name:
        continue
    cases_parsed += 1

    for picture in path.glob('*'):
        image = Image.open(picture).convert('L')  # no rgb
        image.show()

        if image.size != (256, 256):
            image = image.resize((256, 256))

        data = np.array(image, dtype=np.float64).reshape(1, -1) / 255
        data_pairs.append((cases_parsed - 1, data))

nn = NeuralNetwork(NUMBER_INODES, NUMBER_HNODES, cases_parsed, LEARNING_RATE, BASE_DIRECTORY)

all_inputs = []
all_targets = []
random.shuffle(data_pairs)
for category, inputs in data_pairs:
    all_targets.append(get_targets(cases_parsed, category))
    all_inputs.append(inputs)

nn.train_multiple_epochs(all_inputs, all_targets, 1)

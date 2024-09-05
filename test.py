from base import NeuralNetwork
from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIRECTORY = Path('C:\\Plant_leave_diseases_dataset_with_augmentation')

nn = NeuralNetwork(256 * 256 * 1, 100, 4, 0.1, BASE_DIRECTORY)
for i in range(1, 100):
    IMAGE_PATH = BASE_DIRECTORY / 'Blueberry___healthy' / f'image ({i}).jpg'

    image = Image.open(IMAGE_PATH).convert('L')

    if image.size != (256, 256):
        image = image.resize((256, 256))

    data = np.array(image, dtype=np.float64).reshape(1, -1) / 255
    output = nn.query(data).argmax()

    description = sorted([folder for folder in BASE_DIRECTORY.iterdir() if folder.is_dir() and 'Apple' in folder.name])[output]
    print(data,description)

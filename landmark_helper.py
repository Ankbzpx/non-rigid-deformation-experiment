import numpy as np
import json


def read_landmarks(file_path):
    lms_data = np.array(json.load(open(file_path)))

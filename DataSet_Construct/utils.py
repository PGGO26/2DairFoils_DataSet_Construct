import os
import numpy as np
from PIL import Image
from matplotlib import cm

def finalstep(file_dir:str):
    file_lst = os.listdir(file_dir)
    filtered_lst = [x for x in file_lst if x.isdigit()]
    int_lst = sorted([int(item) for item in filtered_lst])
    return int_lst[-1]


def extract_node_coordinates(file_path):
    with open(file_path, 'r') as file:
        nodes_started = False
        coordinates = []

        for line in file:
            if line.startswith("$Nodes"):
                nodes_started = True
            elif line.startswith("$EndNodes"):
                break
            elif nodes_started:
                parts = line.strip().split()
                if len(parts) == 4:
                    coordinates.append(tuple(map(float, parts[1:])))

    return coordinates

def saveAsImage(filename, field_param):
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    field /= max_value

    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)

def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)
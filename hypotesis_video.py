import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
import operator
import os
import cv2

aps  = np.array([])
smartphone  = np.array([])
error_dict = {}


def generate_random_room(width, length,hight):
    """Generate a random room of given width and length."""
    return np.array([random.uniform(0, width), random.uniform(0, length), random.uniform(0, hight)])

def generate_random_aps(num_aps, width, length,hight):
    """Generate a list of random access points in the room."""
    aps = [generate_random_room(width, length,hight) for _ in range(num_aps)]
    return np.array(aps)


def generate_distances(smartphone, aps):
    """Generate distances between smartphone and access points."""
    distances = np.linalg.norm(aps - smartphone, axis=1)
    return distances


def add_noise(distances, std_dev):
    """Add noise to distances."""
    return distances + np.random.normal(0, std_dev, len(distances))


def gradient_error(smartphone, aps, measured_distances):
    """Compute gradient of error between measured and estimated distances."""

    estimated_distances = np.linalg.norm(aps - smartphone, axis=1)
    error = estimated_distances - measured_distances

    gradient = 2 * np.sum((error[:, np.newaxis] * (smartphone - aps)) / estimated_distances[:, np.newaxis], axis=0)
    return gradient



def lms(guess, aps, measured_distances, learning_rate=0.01, tolerance=1e-3):
    """Apply LMS algorithm to find smartphone location."""
    gradient = gradient_error(guess, aps, measured_distances)
    while np.linalg.norm(gradient) > tolerance:
        guess = guess - learning_rate * gradient
        gradient = gradient_error(guess, aps, measured_distances)

    return guess




def process(file_name='EXP_1'):
    initial_guess = np.random.rand(3)

    df = pd.read_csv(f'/home/marco//Documents/WiFiRTT/{file_name}.csv')

    with open('/home/marco/Documents/WiFiRTT/mobile_location.json', "r") as f:
        data = json.load(f)

    exp = [obj for obj in data if obj['name'] == f'{file_name}'][0]

    x = float(exp['xCoordinate'].replace(',', '.'))
    y = float(exp['yCoordinate'].replace(',', '.'))
    z = float(exp['zCoordinate'].replace(',', '.'))

    smartphone = np.array([x, y, z])

    # Create a numpy array from the properties
    smartphone = np.array([x, y, z])

    # Initialize a list to store mean distances
    mean_distances = []

    # Loop through each 'FTM_RESPONDER_X
    responders = df.drop_duplicates(subset='SSID', inplace=False)['SSID']
    APsPlot = df.drop_duplicates(subset='SSID', inplace=False)[
        ['xCoordinate', 'yCoordinate', 'zCoordinate', 'SSID']].values






    combinations = list(itertools.combinations(responders, 3))
    zz = 0;
    #num_rows = len(df)
    num_rows = 30
    for c in combinations:
        for row_index in range(0, num_rows, 10):
            if (row_index + 10 >= num_rows):
                break
            aps = []
            combination_distances = []
            for idx in range(row_index, row_index + 10):
                row = df.iloc[idx]
                ap_location = row[['xCoordinate', 'yCoordinate', 'zCoordinate']].values

                if(row['SSID'] in c):
                    combination_distances.append(row.distance)
                    aps.append(ap_location.astype(float))

            estimated_smartphone = lms(initial_guess, aps, combination_distances)

            width, length, hight = 10, 6, 10
            plt.scatter(smartphone[0], smartphone[1], marker='o', color='blue', label='smartphone')
            # Iterate through all Access Points and adjust the color based on the combination
            for i, ap in enumerate(APsPlot):
                color = 'purple' if ap[-1] in c else 'red'
                plt.scatter(ap[0], ap[1], marker='^', color=color, label='Access Point' if i == 0 else "")
                plt.annotate(ap[-1], (ap[0], ap[1]), xytext=(0, -15), textcoords='offset points', ha='center')

            plt.scatter(estimated_smartphone[0], estimated_smartphone[1], marker='x', color='green')
            plt.xlim(-1, width + 1)
            plt.ylim(-1, length + 1)
            plt.savefig(os.path.join('images_to_turn_into_video', f'img_{zz}.png'))
            plt.clf()
            zz+=1

    generate_video(num_rows, zz)

def generate_video(num_rows,zz):
    img_array = []
    for i in range(0,zz-1):
        try:
            filename = os.path.join('images_to_turn_into_video',f'img_{i}.png')
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        except:
            print(os.path.join('images_to_turn_into_video',f'img_{i}.png'))

    video_name = 'estimated_positions_video.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 10, size)

    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()

    # Remove individual image files
    for i in range(0, zz-1):
            try:
                filename = os.path.join('images_to_turn_into_video', f'img_{i}.png')
                os.remove(filename)
            except:
                pass


if __name__ == '__main__':
    process('EXP_50')
    #print(estimated_smartphone)
    #
def get_error():
    errors =  []
    for i in range(1,53):
        aps, smartphone, estimated_location = process(f'EXP_{i}')
        distance_between_bet_and_real = np.linalg.norm(np.array(estimated_location) - smartphone, axis=0)
        errors.append(distance_between_bet_and_real)
    return errors
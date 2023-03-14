import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import itertools
import operator

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

def plot(estimated_smartphone):
    width, length, hight = 10, 6, 10

    plt.scatter(estimated_smartphone[0], estimated_smartphone[1], marker='x', color='green')
    plt.xlim(-1, width+1)
    plt.ylim(-1, length+1)
    plt.legend()



def process(file_name='EXP_1'):
    initial_guess = np.random.rand(3)

    df = pd.read_csv(f'/home/marco//Documents/WiFiRTT/{file_name}.csv')

    # Load the JSON file
    with open('/home/marco/Documents/WiFiRTT/mobile_location.json', "r") as f:
        data = json.load(f)

    exp = [obj for obj in data if obj['name'] == f'{file_name}'][0]

    # Extract the properties xCoordinate, yCoordinate, zCoordinate
    x = float(exp['xCoordinate'].replace(',', '.'))
    y = float(exp['yCoordinate'].replace(',', '.'))
    z = float(exp['zCoordinate'].replace(',', '.'))

    # Create a numpy array from the properties
    smartphone = np.array([x, y, z])
    plt.scatter(smartphone[0], smartphone[1], marker='o', color='blue',label='smartphone')

    # Initialize a list to store mean distances
    mean_distances = []

    # Loop through each 'FTM_RESPONDER_X
    responders = df.drop_duplicates(subset='SSID', inplace=False)['SSID']
    APsPlot = df.drop_duplicates(subset='SSID', inplace=False)[['xCoordinate', 'yCoordinate', 'zCoordinate','SSID']].values

    plt.scatter(APsPlot[:, 0], APsPlot[:, 1], marker='^', color='red', label='Access Point')
    for i, ssid in enumerate(APsPlot[:, -1]):
        plt.annotate(ssid, (APsPlot[i, 0], APsPlot[i, 1]), xytext=(0, -15), textcoords='offset points', ha='center')

    combinations = list(itertools.combinations(responders, 3))
    print(f'Number of combination {len(combinations)}')
    for c in combinations:
        aps = []
        for responder in c:
            ap_location = df[df['SSID'].isin([str(responder)])][['xCoordinate', 'yCoordinate', 'zCoordinate']].iloc[0]
            aps.append(ap_location.values)
            # Get all the distances corresponding to this responder
            distances = df.loc[df['SSID'] == responder, 'distance'].values
            distances = np.maximum(distances - 1, 0.1)
            # Calculate the mean of distances
            mean_distance = np.mean(distances)

            # Append the mean distance to the list
            mean_distances.append(mean_distance)

        # Convert the list to a numpy array
        mean_distances = np.array(mean_distances)
        estimated_smartphone = lms(initial_guess, aps, mean_distances)
        plot(estimated_smartphone)
        true_distance =  np.linalg.norm(aps - smartphone, axis=1)
        error_dict[str(c)]= np.sum(abs(true_distance - estimated_smartphone))
        print (f'{aps}, {smartphone}, {estimated_smartphone}')
        mean_distances = []
if __name__ == '__main__':
    process('EXP_52')
    plt.savefig('combination_guesses.png')
    plt.show()

    # sort the dictionary by error value in ascending order
    sorted_dict = dict(sorted(error_dict.items(), key=operator.itemgetter(1)))

    # write the top n combinations to a text file
    with open('top_combinations.txt', 'w') as f:
        f.write('Top  Combinations\n')
        f.write('------------------\n')
        f.write('Combination\tError\n')
        for i, (key, value) in enumerate(sorted_dict.items()):

            f.write('%s\t\t%.2f\n' % (str(key), value))


    #print(estimated_smartphone)
    #
def get_error():
    errors =  []
    for i in range(1,53):
        aps, smartphone, estimated_location = process(f'EXP_{i}')
        distance_between_bet_and_real = np.linalg.norm(np.array(estimated_location) - smartphone, axis=0)
        errors.append(distance_between_bet_and_real)
    return errors
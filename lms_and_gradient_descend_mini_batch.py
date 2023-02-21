import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
aps  = np.array([])
smartphone  = np.array([])

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


def gradient_error(smartphone, aps, measured_distances, batch_size=10):
    """Compute gradient of error between measured and estimated distances using mini-batch gradient descent."""
    n_aps = aps.shape[0]
    n_batches = int(np.ceil(n_aps / batch_size))

    gradient_list = []
    for i in range(n_batches):
        start = i * batch_size
        end = min(n_aps, (i + 1) * batch_size)
        batch_aps = aps[start:end, :]
        batch_measured_distances = measured_distances[start:end]

        estimated_distances = np.linalg.norm(batch_aps - smartphone, axis=1)
        error = estimated_distances - batch_measured_distances

        batch_gradient = 2 * np.sum(
            (error[:, np.newaxis] * (smartphone - batch_aps)) / estimated_distances[:, np.newaxis], axis=0)
        gradient_list.append(batch_gradient)

    mean = np.mean(np.vstack(np.array(gradient_list)), axis=0)
    return mean



def lms(guess, aps, measured_distances, learning_rate=0.01, tolerance=1e-3):
    """Apply LMS algorithm to find smartphone location."""
    gradient = gradient_error(guess, aps, measured_distances)
    while np.linalg.norm(gradient) > tolerance:
        guess = guess - learning_rate * gradient
        gradient = gradient_error(guess, aps, measured_distances)

    return guess

def plot(aps,smartphone,estimated_smartphone):
    width, length, hight = 10, 6, 10
    # Plot the room, access points, and smartphone locations
    plt.scatter(aps[:, 0], aps[:, 1], marker='^', color='red', label='Access Point')
    plt.scatter(smartphone[0], smartphone[1], marker='o', color='blue', label='True Smartphone Location')
    plt.scatter(estimated_smartphone[0], estimated_smartphone[1], marker='x', color='green',
                label='Estimated Smartphone Location')
    plt.xlim(-1, width+1)
    plt.ylim(-1, length+1)
    plt.legend()
    plt.show()
def process(file_name='EXP_1'):
    initial_guess = np.random.rand(3)

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
    df = pd.read_csv(f'/home/marco/Documents/WiFiRTT/{file_name}.csv')
    aps = df[['xCoordinate', 'yCoordinate', 'zCoordinate']].values
    measured_distances = df['distance'].to_numpy()
    measured_distances = np.maximum(measured_distances - 1, 0.1)
    estimated_smartphone = lms(initial_guess, aps, measured_distances)
    return aps,smartphone,estimated_smartphone
if __name__ == '__main__':
    aps, smartphone,estimated_smartphone=process('EXP_25')
    print(estimated_smartphone)
    plot(aps,smartphone,estimated_smartphone)
def get_error():
    errors =  []
    for i in range(1,53):
        aps, smartphone, estimated_location = process(f'EXP_{i}')
        distance_between_bet_and_real = np.linalg.norm(np.array(estimated_location) - smartphone, axis=0)
        errors.append(distance_between_bet_and_real)
    return errors
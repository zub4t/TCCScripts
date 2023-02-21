import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
def bayesian_grid_update(aps, measurements):
    """Estimate smartphone location using Bayesian Grid Update."""
    num_aps = aps.shape[0]
    grid = np.zeros((10, 10))
    for i in range(num_aps):
        ap = aps[i, :]
        measurement = measurements[i]
        for x in range(10):
            for y in range(10):
                grid[x, y] += np.exp(-((x - ap[0])**2 + (y - ap[1])**2) / (2 * measurement**2))
    return np.unravel_index(np.argmax(grid), grid.shape)


def process(file_name='EXP_1'):

    df = pd.read_csv(f'/home/marco//Documents/WiFiRTT/{file_name}.csv')

    aps = df[['xCoordinate', 'yCoordinate']].values
    measurements = df['distance'].values
    #measurements = np.maximum(measurements -1, 0.1)

    with open('/home/marco/Documents/WiFiRTT/mobile_location.json', "r") as f:
        data = json.load(f)

    exp = [obj for obj in data if obj['name'] == f'{file_name}'][0]

    # Extract the properties xCoordinate, yCoordinate, zCoordinate
    x = float(exp['xCoordinate'].replace(',', '.'))
    y = float(exp['yCoordinate'].replace(',', '.'))
    z = float(exp['zCoordinate'].replace(',', '.'))

    # Create a numpy array from the properties
    smartphone = np.array([x, y, z])
    # Estimate smartphone location using Bayesian Grid Update
    estimated_location = bayesian_grid_update(aps, measurements)
    return aps,smartphone,estimated_location

if __name__ == '__main__':

    aps, smartphone,estimated_location = process('EXP_17')
    # Plot AP locations, true smartphone location, and estimated smartphone location
    plt.scatter(aps[:, 0], aps[:, 1], marker='^', color='red', label='APs')
    plt.scatter(smartphone[0], smartphone[1],  marker='o', color='blue', label='True Location')
    plt.scatter(estimated_location[0], estimated_location[1], marker='x', color='green', label='Estimated Location')
    plt.legend()
    plt.show()


def get_error():
    errors =  []
    for i in range(1,53):
        aps, smartphone, estimated_location = process(f'EXP_{i}')
        distance_between_bet_and_real = np.linalg.norm(np.array(estimated_location) - smartphone[:2], axis=0)
        errors.append(distance_between_bet_and_real)
    return errors
import tkinter as tk
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

# Define the grid of possible locations
num_cells = 100
grid = np.ones((num_cells, num_cells)) / num_cells**2

# Define the prior belief about the target location

# Define the sensor error
sensor_error = 1

# Generate a random location for the true target
true_location = np.random.randint(0, num_cells, size=(2,))

# Generate random locations for access points (APs)
num_aps = 5
ap_locations = np.random.randint(0, num_cells, size=(num_aps, 2))

# Define the colors for visualization
cmap = plt.cm.Blues
norm = plt.Normalize(vmin=0, vmax=1)

# Define the canvas size and the scale factor
canvas_size = 500
scale_factor = canvas_size / num_cells

# Define the motion model
motion = [0, 0]

# Define the function for drawing the grid and the AP locations
def draw_grid():
    for i in range(num_cells):
        for j in range(num_cells):
            x1 = i * scale_factor
            y1 = j * scale_factor
            x2 = (i + 1) * scale_factor
            y2 = (j + 1) * scale_factor
            canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='gray')
def draw_aps():
    for ap_location in ap_locations:
        x = ap_location[0] * scale_factor + scale_factor / 2
        y = ap_location[1] * scale_factor + scale_factor / 2
        canvas.create_polygon(x, y - 10, x + 10, y + 10, x - 10, y + 10, fill='red')
    if true_location is not None:
        x = true_location[0] * scale_factor + scale_factor / 2
        y = true_location[1] * scale_factor + scale_factor / 2
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='green')


def update_posterior(prior, posterior):
    # Define the sensor model
    sensor_model = np.zeros_like(grid)
    measurement = generate_measurement()
    for a in range(num_aps):
        for i in range(num_cells):
            for j in range(num_cells):
                distance = np.sqrt((i - ap_locations[a][0])**2 + (j - ap_locations[a][1])**2)
                #print(abs(distance-measurement[a]))
                sensor_model[i, j] += np.exp(-0.5 * (distance-measurement[a])**2 / (sensor_error) ** 2)
   # Compute the unnormalized posterior belief by multiplying the prior and sensor models
    posterior_unnormalized = prior * sensor_model
    
    # Normalize the posterior belief to sum to 1
    posterior = posterior_unnormalized / np.sum(posterior_unnormalized)
    
    # Update the plot with the posterior belief
    draw_posterior(posterior)
    canvas.update()
    print(posterior)


        
def draw_posterior(posterior):
    # Clear the canvas
    #canvas.delete('all')

    # Draw the grid and AP locations
    colors = cmap(norm(posterior*500))
    # Loop over all the cells in the grid and draw a colored rectangle for each cell
    for i in range(num_cells):
            for j in range(num_cells):
                x1 = i * scale_factor
                y1 = j * scale_factor
                x2 = (i + 1) * scale_factor
                y2 = (j + 1) * scale_factor
                color = matplotlib.colors.to_hex(colors[i, j])
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
    draw_aps()

        # Update the canvas to show the new colors
    #canvas.update()



# Define the function for generating random measurements
def generate_measurement():
    true_distance = np.sqrt(np.sum((true_location - ap_locations)**2, axis=1))
    measurement = true_distance + np.random.randn(num_aps) * sensor_error
    return measurement





# Create the tkinter canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size)
canvas.pack()

# Call the draw_grid_and_aps function to draw the grid and the AP locations
draw_grid()

# Perform the Bayesian update for a fixed number of iterations
prior = grid
posterior = prior
num_iterations = 1000
for t in range(num_iterations):
    # Generate a new measurement
    

    # Perform the Bayesian update
    update_posterior(prior,posterior)

    # Add a delay to slow down the visualization
    #time.sleep(0.5)

# Start the main loop
root.mainloop()

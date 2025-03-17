# ODEyssey: Visualization and Inspection of Biological Models
Python functions for visualizing and inspecting ordinary differential equations (ODEs) and mathematical models, inspired by Modeling Life by Alan Garfinkel et al., (2017) with examples from biology. Visualizations range from simple vector fields and trajectories to more complex visualizations of inspections of equilibrium points, basins of attraction, bifuractions, oscillations, chaotic behaviour, and some functions for general linear algebra and multivariable systems. Furthermore, there are functions to do simple calculations like Euler's method or a Riemann sum. A full list of functions can be found at the end of the README, or they can be accessed with examples under "examples".

## Installation
To install the package, run:

```bash
pip install git+https://github.com/JorritJS/ODEyssey.git
```

## Example
Here's how you can use the functions. Another version with all the code is available under "/examples/" as a jupyter notebook.

First we make sure that the package is installed. You can check this by running the command after installing it.
```bash
pip --version ODEyssey
```
We can run it as well in a Jupyter notebook by
```python
!pip --version ODEyssey.
```
After ensuring that the package is installed. We can load it with
```python
import ODEyssey as ode
```
### Using the package
Let's explore how to inspect a model of Shark and Tuna populations with our package.
First we give the model some parameters:
```python
# Initial conditions
Shark = 100         # Initial amount of shark
Tuna = 30           # Initial amount of tuna
initial_state = [Shark, Tuna]  # Starting points for x and y, in this case Shark and Tuna populations

# Parameters (adjust these for different dynamics)
b = 0.5   # Tuna birth rate
d = 0.2 # Shark death rate
beta = 0.01  # Probability shark catches a tuna (frequency of successful shark-tuna encounters)
m = 0.5 # Amount of food, so in this case the size of the tuna

parameters = (m, d, beta, b)
```
Then we can start making a model and inspect it. A nice visualization to do is a timeseries:
```python
# Define the differential equations for the timeseries function (so we need y and t)
def shark_tuna_model_timeseries(y, t):
    Shark, Tuna = y
    delta_shark = m * beta * Shark * Tuna - d * Shark  # Predation rate * interaction - natural shark death rate
    delta_tuna = b * Tuna - beta * Shark * Tuna  # Tuna growth - predation rate * interaction
    return [delta_shark, delta_tuna]

ode.timeseries(shark_tuna_model_timeseries, 0, 150, 500, initial_state)
```
![image](https://github.com/user-attachments/assets/8636a3fa-01e6-48a7-bcb8-f0e82df35ca2)

This gives us a good understanding of how the population dynamics work. To gain a broader view of the dynamics, we can plot a vector field:
```python
# First we give the model:
def shark_tuna_model(Shark, Tuna):
    delta_shark = m * beta * Shark * Tuna - d * Shark  # Predation rate * interaction - natural shark death rate
    delta_tuna = b * Tuna - beta * Shark * Tuna  # Tuna growth - predation rate * interaction
    return [delta_shark, delta_tuna]

ode.plot_vector_field(shark_tuna_model, (0, 100), (0, 100), axes_labels=['Shark', 'Tuna'])
```
![image](https://github.com/user-attachments/assets/84d24862-3474-4ce3-9ae5-297186ba0e92)

This is great! Now we might be interested in following a certain state and see where it leads, we can plot a trajectory:
```python
# First we have to give
# Directional vectors
def equations(state, t, m, d, beta, b):
    Shark, Tuna = state
    u = m * beta * Shark * Tuna - d * Shark
    v = b * Tuna - beta * Shark * Tuna
    return[u, v]

ode.plot_trajectories(equations, 0, 40, 0.01, params=parameters)
```
![image](https://github.com/user-attachments/assets/65238a9f-5a75-4c45-88d8-684040bf49dc)

Maybe instead we had a 3-dimensional system we work with. Let's use a Lorenz attractor for this purpose (who doesn't like the beautiful graphics this produces):
```python
# Directional vectors
def lorenz(state, t, sigma, beta, rho):
  x, y, z = state
  dx = sigma * (y - x)
  dy = x * (rho - z) - y
  dz = x * y - beta * z
  return [dx, dy, dz]

# Parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

p = (sigma, beta, rho)

initial_state = [1.0, 1.0, 1.0]

ode.plot_trajectories_3d(lorenz, 0, 40, 0.01, initial_state=initial_state, params=p)
```
![image](https://github.com/user-attachments/assets/b544e4fc-7f0a-431b-aa3b-1ff97670f410)

## Functions available (more to come):
- Euler's method
- Riemann sum
- Vector field plots (2D and 3D)
- Trajetory plots (2D and 3D)
- Timeseries (with optional interventions)

All plots can be customized for different styles.
Examples are given in the function descriptions (accessible through help(ode.functionName)), and a Jupyter notebook is available under "/examples/".

## Contributing
Contributions are welcome! Please raise an issue or fork this repository and submit pull requests to add new features or fix bugs.

## License
This project is licensed under the MIT License.

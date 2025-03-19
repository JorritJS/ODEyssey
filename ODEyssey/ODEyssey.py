# Import required modules
from typing import Callable, Tuple, List, Dict
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import inspect
from sympy import symbols, diff, solve, lambdify


def eulers_method(
        x_start: float,
        x_end: float,
        step_size: float,
        equation: Callable[[float], float],
        differential_equation: Callable[[float], float],
        plot_trajectory: bool = False,
        verbose: bool = False,
        dark_mode: bool = True
    ) -> Tuple[list[float], list[float]]:
    """
    This function performs Euler's method calculation.

    Args:
        x_start (float): Starting value of x.
        x_end (float): End value of x.
        step_size (float): Size of each step.
        equation (Callable[[float], float]): Function representing the equation.
        differential_equation (Callable[[float], float]): Function representing the differential equation.
        plot_trajectory (bool, optional): If True, plot the trajectory. Defaults to False.
        verbose (bool, optional): If True, prints intermediate values. Defaults to False.
        dark_mode (bool, optional): If True, set the plot style to dark. Defaults to True.
        
    Returns:
        Tuple[List[float], List[float]]: Two lists containing x and y values from Euler's method.
        
    Example:
        def equation(x):
            return x**2
        
        def differential_equation(x):
            return 2*x
        
        x_values, y_values = eulers_method(0, 5, 0.1, equation, differential_equation, plot_trajectory = True, verbose = True)
    """
    
    if step_size <= 0:
        raise ValueError('Step size must be above 0')
    
    if not callable(equation) or not callable(differential_equation):
        raise TypeError('Both equation and differential equation must be callable')

    # The number of steps we use in our calculation
    steps = int((x_end - x_start) / step_size)

    # Starting values of both x and y
    x = x_start
    y = equation(x_start) # Calculate with the equation since the value of y is function(x) where function(x) is our equation

    # Initialize lists for output values, the first value we give since it will not be added in the for loop
    solution_x_values = [x_start]
    solution_y_values = [equation(x_start)]

    # Here we calculate our values by going over all the values in a step-wise manner
    for _ in range(steps):
        x = x + step_size # Update the x value; can be x += step_size but for clarity left like this
        y = y + step_size * differential_equation(x) # Calculate our y value

        # Add them to the lists
        solution_x_values.append(x)
        solution_y_values.append(y)

        if verbose:
            print(f"x: {x:.4f}, y: {y:.6f}")

    # Plot the trajectory if we want it
    if plot_trajectory:
        # Set plot style, white or black
        if dark_mode == True:
            plt.style.use('dark_background')
            title_color = 'white'
            color = 'white'
        else:
            plt.style.use('default')
            title_color = 'black'
            color = 'black'
        
        plt.plot(solution_x_values, solution_y_values, marker='o', linestyle='-', color=color, label='Euler\'s Method')
        plt.title('Trajectory of y as a Function of x Using Euler\'s Method', color = title_color)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()

    return solution_x_values, solution_y_values


def plot_vector_field(
        vector_field_functions: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], # maybe numpy array
        x_grid: Tuple[float, float],
        y_grid: Tuple[float, float],
        plot_points: int = 20,
        axes_labels: Tuple[str, str] = ("x", "y"),
        gradient: bool = True,
        grid: bool = False,
        dark_mode: bool = True,
        title: str = "",
        show: bool = True
    ) -> None:
    '''
    This function plots a vector field (2D) for two given equations.

    Args:
        equations Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Function representing the two equations.
        x_grid (Tuple[float, float]): Start and end values for the grid on the x-axis.
        y_grid (Tuple[float, float]): Start and end values for the grid on the y-axis.
        vector_colour (str, optional): If given, use that colour. Defaults to "black".
        plot_points(int, optional): If given, the number of vectors plotted. Defaults to 20 points.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        gradient (bool, optional): boolean statement determining whether a gradient of the vector size is shown or not. Defaults to True.
        grid (bool, optional): boolean statement determining whether a grid is shown or not. Defaults to False.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (ggplot style). Defaults to True.
        title (str, optional): If given, the plot title. Defaults to emtpy ("").
        show (bool, optional): If True then show the plot, otherwise don't show it. This is used since the function is called in
            other functions as well, giving the necessity for this option. Defaults to True.
        
    Returns:
        plt: A plot with the 2D vector field.
   
    Example:
        def equations(x , y):
            u = -y
            v = x
            return u, v

        plot_vector_field(equations, (-2, 2), (-2, 2))
    '''
    try:
        # Grid
        x, y = np.meshgrid(np.linspace(x_grid[0], x_grid[1], plot_points),
                        np.linspace(y_grid[0], y_grid[1], plot_points))

        # Equations to plot
        u, v = vector_field_functions(x, y)

        # Set plot style
        plt.style.use('dark_background' if dark_mode else 'default')
        title_color = 'white' if dark_mode else 'black'

        # To still be able to see the vectors with these exact conditions
        if gradient:
            vector_colour = 'black' if dark_mode else 'white'
        else:
            vector_colour = 'white' if dark_mode else 'black'

        # Gradient plot
        if gradient == True:
            c = plt.imshow(abs(u) + abs(v), extent=(x_grid[0], x_grid[1], y_grid[0], y_grid[1]), interpolation='none', origin='lower')
            colorbar = plt.colorbar(c)
            colorbar.set_label('Relative length of vector', rotation=270, labelpad = 20)

        # Plot
        plt.quiver(x, y, u, v, color = vector_colour)
        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])

        # Title, if given then use that otherwise automatic one
        if title:
            plt.title(title, pad = 15, color = title_color)
        else:
            title = f"2D vector field superimposed on {axes_labels[0]}-{axes_labels[1]} state space"
            plt.title(title, pad = 15, color = title_color)
        
        #plt.title(title, pad = 15, color = title_color)
        plt.grid(grid) # Plots the grid if set to True
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        
        if show == True:
            plt.show()
    except Exception as e:
        print(f'An error occured while plotting the vector field: {e}')


def plot_3d_vector_field(
        vector_field_functions: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        x_grid: Tuple[float, float],
        y_grid: Tuple[float, float],
        z_grid: Tuple[float, float],
        plot_points: int = 10,
        dark_mode: bool = True,
        gradient: bool = True,
        grid: bool = False,
        axes_labels: Tuple[str, str] = ("x", "y", "z"),
        vector_scale_factor: float = 1,
        title: str = ""
    ) -> None:
    '''
    This function plots a 3D vector field for given vector field equations.

    Args:
        vector_field_functions (Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]): 
            Function representing the vector field equations.
        x_grid (Tuple[float, float]): Start and end values for the grid on the x-axis.
        y_grid (Tuple[float, float]): Start and end values for the grid on the y-axis.
        z_grid (Tuple[float, float]): Start and end values for the grid on the z-axis.
        plot_points (int, optional): If given, the number of points in each dimension. Defaults to 10.
        dark_mode (bool, optional): Boolean statement determining whether the plot style is dark or not. Defaults to True.
        gradient (bool, optional): Boolean statement determining whether a gradient of the vector size is shown or not. Defaults to True.
        grid (bool, optional): Boolean statement determining whether a grid is shown or not. Defaults to False.
        axes_labels (Tuple[str, str, str], optional): If given, the axes labels for the x, y and z axes. Defaults to "x", "y", and "z".
        vector_scale_factor (float, optional): scales the vector arrows. Defaults to 1.        
        title (str, optional): If given, the plot title. Defaults to empty ("").
        
    Returns:
        None: Displays a plot with the 3D vector field.
   
    Example:
        def equations(x, y, z):
            u = x
            v = y
            w = z
            return u, v, w

        plot_3d_vector_field(equations, (-2, 2), (-2, 2), (-2, 2), vector_scale_factor=0.5)
    '''
    try:
        # Create a grid
        x, y, z = np.meshgrid(np.linspace(x_grid[0], x_grid[1], plot_points),
                               np.linspace(y_grid[0], y_grid[1], plot_points),
                               np.linspace(z_grid[0], z_grid[1], plot_points))

        # Get vector components
        u, v, w = vector_field_functions(x, y, z)

        # Calculate the magnitude of the vectors
        M = np.sqrt(u**2 + v**2 + w**2)

        # Normalize the magnitudes
        minimum = np.min(M)
        maximum = np.max(M)
        normalized = (M - minimum) / (maximum - minimum)

        # Set plot style
        if dark_mode:
            plt.style.use('dark_background')
            title_color = 'white'
        else:
            plt.style.use('default')
            title_color = 'black'

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Gradient plot
        if gradient:
            c = ax.scatter(x, y, z, c=normalized, cmap='viridis', marker='o', alpha=0.5)
            fig.colorbar(c, ax=ax, label='Magnitude of vector')

        # Colors based on magnitudes
        colors = cm.viridis(normalized)
        colors = np.reshape(colors, [-1, 4])

        # Plot the quiver plot
        ax.quiver(x, y, z, u, v, w, color=colors, length=0.2 * vector_scale_factor)

        # Set plot labels and title
        if title:
            ax.set_title(title, pad=15, color=title_color)
        else:
            ax.set_title('', pad=15, color=title_color)

        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_zlabel(axes_labels[2])

        # These remove the background color of the plot axes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Set grid if specified
        ax.grid(grid)

        plt.show()
    except Exception as e:
        print(f'An error occurred while plotting the 3D vector field: {e}')


def plot_trajectories(
        equations: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        start: float,
        stop: float,
        step_size: float,
        vector_field: bool = True,
        params: list = [],
        axes_labels: Tuple[str, str] = ('x','y'),
        title: str = '',
        grid: bool = False,
        title_color: str = None,
        dark_mode: bool = True,
        gradient: bool = True,
        gradient_trajectory: bool = True,
        **kwargs
        ) -> None:
    '''
    This function plots a trajectory and optional vector field (2D) for two given equations.

    Args:
        equations Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Function representing the two equations.
        start (float): starting value for the trajectory (time value).
        stop (float): end value for the trajectory (time value).
        step_size (float): size of the step we take for odeint function (comparable to Euler's step size)
        vector_field (bool, optional): sets t.
        params (list, optional):
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        title (str, optional): If given, the plot title. Defaults to emtpy ("").
        grid (bool, optional): boolean statement determining whether a grid is shown or not. Defaults to False.
        title_color (str, optional): determines the colour of the title. Defaults to black, certain settings automatically change it for readability.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (default style). Defaults to True.
        gradient (bool, optional): boolean statement determining whether a gradient of the vector size is shown or not. Defaults to True.
        gradient_trajectory (bool, optional): boolean statement determining whether the trajectory is colored according to a gradient. Defaults to True..
        
    Returns:
        plt: A plot with the 2D trajectory and an optional vector field.
   
    Example:
        # Directional vectors
        def equations(state, t, m, d, beta, b):
            Shark, Tuna = state
            u = m * beta * Shark * Tuna - d * Shark
            v = b * Tuna - beta * Shark * Tuna
            return[u, v]

        # Parameters (adjust these for different dynamics)
        b = 0.5
        d = 0.2
        beta = 0.01
        m = 0.5

        p = (m, d, beta, b)

        initial_state = [100, 30] # Starting points for x and y, in this case Shark and Tuna populations

        plot_trajectories(equations, 0, 40, 0.01, params=p)
    '''
    try:
        # If gradient_trajectory is set to True, gradient is turned off for readability
        if gradient_trajectory and gradient:
            gradient = False
            print('Gradient of state space turned off for readability')

        # Calculate the trajectory
        t = np.arange(start, stop, step_size)
        result = odeint(equations, initial_state, t, args=tuple(params))

        # Set plot style
        plt.style.use('dark_background' if dark_mode else 'default')
        if title_color is None:
            title_color = 'white' if dark_mode else 'black'
        
        if gradient:
            colour = 'white'
        else:
            colour = 'white' if dark_mode else 'black'

        # Optional vector field, taking the dimensions of the trajectory as the x_grid and y_grid. Show is set to False in plot_vector_field to prevent multiple outputs. 
        if vector_field == True:        
            def vector_field_func(x, y):
                return equations([x, y], 0, *params)

            try:
                plot_vector_field(vector_field_func,
                                x_grid=(min(result[:, 0])-max(result[:, 0])*0.05, max(result[:, 0])*1.05),
                                y_grid=(min(result[:, 1])-max(result[:, 1]*0.05), max(result[:, 1])*1.05),
                                show = False,
                                gradient = gradient,
                                dark_mode = dark_mode,
                                grid = grid,
                                )
            except Exception as e:
                print(f'Something went wrong while plotting the vector field: {e}')

        # Plot the trajectory
        if gradient_trajectory:
            # Plot the trajectory with changing colors
            normalized = plt.Normalize(t.min(), t.max())
            colors = cm.viridis(normalized(t))
            for i in range(len(t) - 1):
                plt.plot(result[i:i+2, 0], result[i:i+2, 1], color = colors[i])
        else:
            plt.plot(result[:, 0], result[:, 1], color = colour)

        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1]) 
        plt.title(title, pad = 15, color = title_color)
        plt.grid(grid) # Plots the grid if set to True
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.show()
    except Exception as e:
        print(f'An error occured while plotting the trajectories: {e}')


def plot_trajectories_3d(
    equations: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    start: float,
    stop: float,
    step_size: float,
    initial_state: list,
    vector_field: bool = True,
    params: list = [],
    axes_labels: Tuple[str, str, str] = ('x','y','z'),
    title: str = '',
    grid: bool = False,
    title_color: str = None,
    dark_mode: bool = True,
    gradient: bool = True,
    gradient_trajectory: bool = True,
    vector_scale_factor: float = 1,
    colorbar: bool = True,
    trajectory_colour: str = 'salmon',
    **kwargs
    ) -> None:
    """
    This function plots a trajectory and optional vector field (3D) for three given equations.

    Args:
        equations (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): Function representing the two equations.
        start (float): starting value for the trajectory (time value).
        stop (float): end value for the trajectory (time value).
        step_size (float): size of the step we take for odeint function (comparable to Euler's step size).
        initial_state (list): where the trajectory starts.
        vector_field (bool, optional): sets t.
        params (list, optional): the parameters that are needed for the functions.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        title (str, optional): If given, the plot title. Defaults to emtpy ("").
        grid (bool, optional): boolean statement determining whether a grid is shown or not. Defaults to False.
        title_color (str, optional): determines the colour of the title. Defaults to black, certain settings automatically change it for readability.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (default style). Defaults to True.
        gradient (bool, optional): boolean statement determining whether a gradient of the vector size is shown or not. Defaults to True.
        gradient_trajectory (bool, optional): boolean statement determining whether the trajectory is colored according to a gradient. Defaults to True.
        vector_scale_factor (float, optional): scales the vector arrows. Defaults to 1.
        colorbar (bool, optional): whether or not we get a colorbar in our plot. Defaults to True.
        trajectory_colour (str, optional): the colour of the trajectory if the trajectory gradient is off. Defaults to "salmon".
        
    Returns:
        plt: A plot with the 3D trajectory and an optional vector field.
   
    Example:
        # Example usage
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

        # Starting position
        initial_state = [1.0, 1.0, 1.0]

        plot_trajectories_3d(lorenz, 0, 40, 0.01, initial_state=initial_state, params=p, vector_scale_factor=1)
    """
    try:
        if gradient_trajectory and gradient:
            gradient = False
            print('Gradient of state space turned off for readability.')

        if trajectory_colour:
            gradient_trajectory = False
            print('Turned of gradient trajectory, cannot have both colour and gradient.')

        t = np.arange(start, stop, step_size)
        result = odeint(equations, initial_state, t, args=tuple(params))

        plt.style.use('dark_background' if dark_mode else 'default')
        if title_color is None:
            title_color = 'white' if dark_mode else 'black'
        colour = 'white' if not dark_mode else 'black'

        fig = plt.figure(facecolor=colour, figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        if gradient_trajectory:
            normalized = plt.Normalize(t.min(), t.max())
            colors = cm.viridis(normalized(t))
            for i in range(len(t) - 1):
                ax.plot(result[i:i+2, 0], result[i:i+2, 1], result[i:i+2, 2], color=colors[i])
        else:
            ax.plot(result[:, 0], result[:, 1], result[:, 2], color=trajectory_colour)

        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        ax.set_zlabel(axes_labels[2])
        ax.grid(grid)
        ax.set_aspect('equal')

        # Calculate individual axis min/max (trajectory bounds)
        x_min, x_max = np.min(result[:, 0]), np.max(result[:, 0])
        y_min, y_max = np.min(result[:, 1]), np.max(result[:, 1])
        z_min, z_max = np.min(result[:, 2]), np.max(result[:, 2])

        # Set the aspect ratio
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        if vector_field:
            # Vector field grid matching trajectory bounds
            x, y, z = np.meshgrid(np.linspace(x_min, x_max, 10),
                                np.linspace(y_min, y_max, 10),
                                np.linspace(z_min, z_max, 10))

            u, v, w = equations([x, y, z], 0, *params)

            if gradient_trajectory is False:
                M = np.sqrt(u**2 + v**2 + w**2)
                norm = plt.Normalize(vmin=np.min(M), vmax=np.max(M))
                colors = cm.viridis(norm(M))
                colors = colors.reshape(-1, 4)

                # Plot the colorbar
                mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
                mappable.set_array(M)
                colorbar = plt.colorbar(mappable, ax=ax)
                colorbar.set_label('Relative vector length', rotation=270, labelpad=15, color = title_color)
            else:
                if dark_mode:
                    colors = 'white'
                else:
                    colors = 'black'

            ax.quiver(x, y, z, u, v, w, color=colors, length=vector_scale_factor, normalize=True)

        # Colorbar for trajectory gradient
        if colorbar and gradient_trajectory:
            mappable = cm.ScalarMappable(norm=normalized, cmap=cm.viridis)
            mappable.set_array(t)
            colorbar = plt.colorbar(mappable, ax=ax)
            colorbar.set_label('Time', rotation=270, labelpad=15, color = title_color)

        #plt.tight_layout()
        plt.title(title, color = title_color)
        plt.show()
    except Exception as e:
        print(f'An error occured while plotting the trajectories: {e}')


def timeseries(
    functions: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_start: float,
    x_end: float,
    n_step: float,
    initial_state: List[float],
    intervention: List[Dict[str, float]] = None,
    title: str = '',
    axes_labels: Tuple[str, str] = ("Time", "Values"),
    line_colors: Tuple[str] = None, #('#fde725', '#5ec962'), # Chnage this from Tuple to list or something.
    grid: bool = False,
    dark_mode: bool = True,
    labels: Tuple[str] = None
    ) -> None:
    """
    This function plots a timeseries of n equations. Optional interventions can be given in a dictionary, as per example.

    Args:
        functions (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): Function representing the equations.
        X_start (float): starting value for the time series (time value, i.e. the starting value on the left side of the plot).
        x_stop (float): end value for the trajectory (time value, i.e. the end value on the right side of the plot).
        n_step (float): size of the step we take for odeint function (comparable to Euler's step size)
        initial_state (List[float]): Initial state vector.
        interventions (List[Dict[str, float]], optional): List of interventions. Each intervention is a dictionary with 'time', 'type', and 'magnitude'.
        title (str, optional): If given, the plot title. Defaults to emtpy ("").
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        line_colors (List[str], optional): Colors for the lines. Defaults to Viridis colours.
        grid (bool, optional): boolean statement determining whether a grid is shown. Defaults to True.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (default style). Defaults to True.

    Returns:
        plt: A plot with the timeseries of the given equations

    Example:
        # Initial conditions
        Shark = 100         # Initial amount of shark
        Tuna = 30           # Initial amount of tuna
        intital_state = [Shark, Tuna]  # The initial state vector

        # Define the change equations
        def shark_tuna_model(y, t):
        Shark, Tuna = y
        delta_shark = m * beta * Shark * Tuna - d * Shark  # Predation rate * interaction - natural shark death rate
        delta_tuna = b * Tuna - beta * Shark * Tuna  # Tuna growth - predation rate * interaction
        return [delta_shark, delta_tuna]

        # Parameters (adjust these for different dynamics)
        b = 0.5   # Tuna birth rate
        d = 0.2 # Shark death rate
        beta = 0.01  # Probability shark catches a tuna (frequency of successful shark-tuna encounters)
        m = 0.5 # Amount of food, so in this case the size of the tuna

        # Interventions
        interventions = [
            {'time': 70, 'type': 'multiply', 'magnitude': 0.5},
            {'time': 90, 'type': 'add', 'magnitude': 10},
            {'time': 110, 'type': 'subtract', 'magnitude': 5},
        ]

        # Plotting the timeseries with interventions
        timeseries(shark_tuna_model, 0, 150, 500, intital_state, interventions)
    """
    try:
        # The amount of time, i.e. the x-axis
        t = np.linspace(x_start, x_end, n_step)

        if intervention:
            # Initialize the values
            solution = []
            current_state = initial_state
            previous_time = x_start

            for inter in intervention:
                # Get the data for the intervention, going over each intervention one by one
                time = inter['time']
                intervention_type = inter['type']
                magnitude = inter['magnitude']

                # Calculate the values up to the first intervention point
                t_up_to_intervention = t[(t >= previous_time) & (t <= time)] # We get the time values from the previous time until the intervention

                # Solve up to the intervention time
                if len(t_up_to_intervention) > 1:
                    solution_up_to_timepoint = odeint(functions, current_state, t_up_to_intervention)
                    solution.append(solution_up_to_timepoint)

                    # Apply the intervention
                    if intervention_type == 'multiply':
                        current_state = [val * magnitude for val in solution_up_to_timepoint[-1]]
                    elif intervention_type == 'add':
                        current_state = [val + magnitude for val in solution_up_to_timepoint[-1]]
                    elif intervention_type == 'subtract':
                        current_state = [val - magnitude for val in solution_up_to_timepoint[-1]]

                    previous_time = time
                else:
                    print('Next intervention is too close to the previous one, add more steps by increasing n_step or change the intervention times.')
                
            # Handle the time points after the last intervention
            t_after_last_intervention = t[t > previous_time]
            if len(t_after_last_intervention) > 1: # Otherwise it cannot calculate
                solution_after_last_intervention = odeint(functions, current_state, t_after_last_intervention)
                solution.append(solution_after_last_intervention)
            else:
                print('Last intervention is too close to the end, add more steps by increasing n_step, or increase x_end.')

            solution = np.vstack(solution)
        else:
            solution = odeint(functions, initial_state, t)

        # Set plot style
        plt.style.use('dark_background' if dark_mode else 'default')
        title_color = 'white' if dark_mode else 'black'

        # Title, if given then use that otherwise automatic one
        if title:
            plt.title(title, pad = 15, color = title_color)
        else:
            title = f"Timeseries of {functions.__name__}"
            plt.title(title, pad = 15, color = title_color)

        # Set the line colors if not provided
        if line_colors is None:
            line_colors = [ "red",
                            "blue",
                            "lightgreen",
                            "yellow",
                            "cyan",
                            "magenta",
                            "orange",
                            "pink",
                            "purple"
            ]

        # Get the labels for plotting if not given
        if labels is None:
            source = inspect.getsource(functions)
            first_line = source.split('\n')[1]
            state_names = first_line.split('=')[0].strip().split(', ')
        
        # Make the plot
        for i in range(solution.shape[1]):
            plt.plot(t, solution[:, i], color=line_colors[i], linestyle='solid', linewidth=2.0, label=state_names[i])

        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        plt.legend()
        plt.grid(grid)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f'An error occurred while plotting the timeseries: {e}')


def riemann(x_start: float,
            x_end: float,
            step_size: float,
            equation: Callable[[float], float],
            method='left',
            dark_mode: bool = False,
            plot: bool = True,
            line_color: str = None,
            bar_color: str = 'skyblue',
            axes_labels: Tuple[str, str] = ("x", "y")
    ) -> float:
    '''
    This function performs a Riemann sum.

    Args:
        x_start (float): Starting value of x.
        x_end (float): End value of x.
        step_size (float): Size of each step.
        equation (Callable[[float], float]): Function representing the equation.
        method (str, optional): If given, the method for the Riemann sum:
            - 'left': Uses the function value at the left end of each subinterval.
            - 'right': Uses the function value at the right end of each subinterval.
            - 'middle': Uses the function value at the midpoint of each subinterval.
            - 'upper': Uses the maximum absolute function value within each subinterval.
            - 'lower': Uses the minimum absolute function value within each subinterval.
        dark_mode (bool, optional): If True, use dark background for the plot.
        plot (bool, optional): If True, plot the Riemann sum.
        line_color (str, optional): Color of the function line in the plot.
        bar_color (str, optional): Color of the bars representing the Riemann sum.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".

    Returns:
        float: The calculated Riemann sum.
        
    Example:
        # Example usage
        def equation(i):
            return np.sin(i)

        result = riemann(0, 10, 0.5, equation, method='upper')
        print(result)
    '''

    # Checking for errors
    if step_size <= 0:
        raise ValueError('Step size must be above 0')
    
    if not callable(equation):
        raise TypeError('Equation must be callable')

    # Calculate the number of steps
    steps = int((x_end - x_start) / step_size)
    
    # Initialize
    total_sum = 0
    xfirst = x_start
    x_values = []
    y_values = []

    # Calculate the sum based on the chosen method
    if method == 'left':
        for step in range(steps):
            y0 = equation(xfirst)
            xsecond = xfirst + step_size
            total_sum += y0 * (xsecond - xfirst)
            x_values.append((xfirst, xsecond))
            y_values.append(y0)
            xfirst = xsecond

    elif method == 'right':
        for step in range(steps):
            xsecond = xfirst + step_size
            y1 = equation(xsecond)
            total_sum += y1 * (xsecond - xfirst)
            x_values.append((xfirst, xsecond))
            y_values.append(y1)
            xfirst = xsecond

    elif method == 'middle':
        for step in range(steps):
            mid = xfirst + step_size / 2
            ymid = equation(mid)
            xsecond = xfirst + step_size
            total_sum += ymid * (xsecond - xfirst)
            x_values.append((xfirst, xsecond))
            y_values.append(ymid)
            xfirst = xsecond

    elif method == 'upper':
        for step in range(steps):
            xsecond = xfirst + step_size

            # To get the maximum value we check which x gets us the highest value
            points = np.linspace(xfirst, xsecond, 1000)
            abs_values = np.abs(equation(points))
            max_abs_value = max(abs_values)

            # For plotting we need the actual value, not the absolute
            max_index = np.argmax(abs_values)
            max_x = points[max_index]
            append_y = equation(max_x)

            # Update the sum and append x and y values for plotting
            total_sum += max_abs_value * (xsecond - xfirst)
            x_values.append((xfirst, xsecond))
            y_values.append(append_y)
            
            # Move to the next subinterval
            xfirst = xsecond
        
    elif method == 'lower':
        for step in range(steps):
            xsecond = xfirst + step_size

            # To get the minimum value we check which x gets us the lowest value
            points = np.linspace(xfirst, xsecond, 500)
            abs_values = np.abs(equation(points))
            min_abs_value = min(abs_values)

            # For plotting we need the actual value, not the absolute
            min_index = np.argmin(abs_values)
            min_x = points[min_index]
            append_y = equation(min_x)

            # Update the sum and append x and y values for plotting
            total_sum += min_abs_value * (xsecond - xfirst)
            x_values.append((xfirst, xsecond))
            y_values.append(append_y)
            
            # Move to the next subinterval
            xfirst = xsecond

    else:
        raise ValueError('Invalid method. Choose from "left", "right", "middle", "upper", or "lower".')

    # Plot the graph if set to 'True'
    if plot:
        try:
            if dark_mode:
                plt.style.use('dark_background')
                color = 'white'
                if line_color is None:
                    line_color = 'white'
                box_background, edgecolor = 'black', 'white'
            else:
                plt.style.use('default')
                color = 'black'
                if line_color is None:
                    line_color = 'black'
                box_background, edgecolor = 'white', 'black'

            x = np.linspace(x_start, x_end, 1000)
            plt.plot(x, equation(x), color=line_color)
            
            for (x1, x2), y in zip(x_values, y_values):
                # Use the original function value for plotting
                plt.bar(x1, y, width=(x2 - x1), align='edge', alpha=1, color = bar_color, edgecolor=color, linewidth=0.5)
            
            plt.xlabel(axes_labels[0])
            plt.ylabel(axes_labels[1])
            plt.title('Riemann Sum Approximation', color=color)
            
            # Add the Riemann sum to the plot
            plt.text(x_start, max(y_values) * 0.9, f'Sum: {total_sum:.4f}', bbox=dict(facecolor=box_background, alpha=0.9, edgecolor=edgecolor, boxstyle='round'), fontsize=12, color=color)
            
            plt.show()
        except Exception as e:
                print(f'An error occured while plotting the Riemann sum graph: {e}')
    
    return total_sum


def linear_stability_analysis(
    differential_equation: Callable[[np.ndarray], np.ndarray],
    variable: str,
    plot: bool = True,
    axes_labels: Tuple[str, str] = ("X", "f(X)"),
    tangent: bool = True,
    arrow_head_size: float = 20,
    dark_mode: bool = True,
    title: str = None,
    ) -> Tuple:
    """
    This function performs a linear stability analysis of a given differential equation by calculating its equilibrium points,
    analyzing their stability, and optionally plotting the results.

    Args:
        differential_equation (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): Function representing the differential equation.
        variable (str): the variable with respect to which the equilibrium points are calculated.
        plot (bool, optional): Determines whether to plot or not. Defaults to True.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "X" and "f(X)".
        tangent (bool, optional): If True, plots the tangent lines at equilibrium points. Defaults to True.
        arrow_head_size (float, optional): Size of the arrowheads in the plot. Defaults to 20.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (default style). Defaults to True.
        title (str, optional): If given, the plot title. Defaults to None.
        
    Returns:
        Tuple: A tuple containing:
            - list: A list of the coordinates of the equilibrium points.
            - plt: The plot of the linear stability analysis (if plotting is set to True).

        list: A list with the coordinates of the equilibrium points.
        plt: A plot with the 3D trajectory and an optional vector field.
   
    Example:
        X = symbols('X')  # Symbolic variable (be sure to call symbols(), otherwise it won't work)
        differential_eq = 2 * X*(1-X/5)*(X/1 -1)  # Example dX/dt
        linear_stability_analysis(differential_eq, X)
    """
    try:
        # Solve df(X)/dX (variables can change, example function)
        differentiated_function = diff(differential_equation, variable)
        print(differentiated_function)

        # Find the equilibrium points
        EP = solve(differential_equation, variable)

        # Filter to keep only real equilibrium points
        real_EP = [point.evalf() for point in EP if point.is_real]

        # Analyze stability at each equilibrium point
        stability_info = []
        differential_func_numeric = lambdify(variable, differentiated_function)
        for point in real_EP:
            slope = differential_func_numeric(float(point))
            stability = "Stable" if slope < 0 else "Unstable" if slope > 0 else "Semi-stable"
            stability_info.append((point, slope, stability))
            print(f"Equilibrium point {point}: Slope = {slope}, Stability = {stability}")

        if plot:
            try:
                # Set dark_mode
                if dark_mode:
                    plt.style.use('dark_background')
                    color = 'white'
                else:
                    plt.style.use('default')
                    color = 'black'

                # Set figure size
                plt.figure(figsize=(10, 6))

                # Calculate a bezel and min and max values to dynamically account for plot-size
                plotting_bezel = float(abs(max(real_EP) - min(real_EP)) * 0.05)
                min_x = float(min(real_EP) - plotting_bezel)
                max_x = float(max(real_EP) + plotting_bezel)

                # Calculate the x-values to plot
                if len(real_EP) > 1:
                    x = np.linspace(min_x, max_x, 1000)
                else:
                    x = np.linspace(-0.5, 0.5, 1000)

                # Derivative values
                differential_equation_num = lambdify(variable, differential_equation)
                x_prime = differential_equation_num(x)
                
                # Plot the line first
                plt.plot(x, x_prime, color=color, linewidth=2, zorder=1)
                plt.hlines(0, min_x - 0.5, max_x + 0.5, color=color, linewidth=0.5, zorder=1)
                if min_x <= 0:
                    plt.axvline(x=0, color=color, linewidth=0.5, zorder=1)

                # Plot equilibrium points and their stability
                for point, slope, stability in stability_info:
                    if dark_mode:
                        plt.scatter([float(point)], 0, color="white" if stability == "Stable" else "black" if stability == "Unstable" else "red", 
                                    label=stability, edgecolors='white', zorder=3)
                    else:
                        plt.scatter([float(point)], 0, color="black" if stability == "Stable" else "white" if stability == "Unstable" else "red", 
                                    label=stability, edgecolors='black', zorder=3)

                    # Plot tangent line
                    if tangent:
                        x_tangent = np.linspace(float(point) - plotting_bezel*2, float(point) + plotting_bezel*2, 1000)
                        tangent_line = slope * (x_tangent - float(point))
                        plt.plot(x_tangent, tangent_line, linestyle="solid", color="red", linewidth=2, zorder=2)

                # Maximum for vector length
                max_arrow_length = (max_x - min_x) * 0.1

                # Sets a default arrow length to ensure right plotting with one EP
                arrow_length = 0.1

                # Plot the vectors                
                for i in range(len(real_EP)-1):
                    point1 = float(real_EP[i])
                    point2 = float(real_EP[i+1])
                    midpoint = (point1 + point2) / 2
                    vector_size = differential_equation_num(midpoint)
                    x_difference = abs(point2 - point1)

                    # Arrow length
                    if len(real_EP) > 1:
                        arrow_length = x_difference * 0.25

                    # Plot the arrows
                    if vector_size > 0:
                        plt.annotate('', 
                                    xy=(midpoint + arrow_length / 2, 0), 
                                    xytext=(midpoint - arrow_length / 2, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                                                    shrinkA=0, shrinkB=0, 
                                                    mutation_scale=arrow_head_size))
                    elif vector_size < 0:
                        plt.annotate('', 
                                    xy=(midpoint - arrow_length / 2, 0), 
                                    xytext=(midpoint + arrow_length / 2, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                                                    shrinkA=0, shrinkB=0, 
                                                    mutation_scale=arrow_head_size))

                # Plot the outermost vectors
                if len(real_EP) > 1:
                    first_point = float(real_EP[0])
                    first_vector_size = differential_equation_num(first_point - 0.1)
                    if first_vector_size > 0:
                        plt.annotate('',
                                    xy=(first_point - 0.5, 0), 
                                    xytext=(first_point - 0.5 - arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                                                    shrinkA=0, shrinkB=0, 
                                                    mutation_scale=arrow_head_size))
                    if first_vector_size < 0:
                        plt.annotate('',
                                    xy=(first_point - 1, 0), 
                                    xytext=(first_point - 1 + arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                                                    shrinkA=0, shrinkB=0, 
                                                    mutation_scale=arrow_head_size))

                    last_point = float(real_EP[-1])
                    last_vector_size = differential_equation_num(last_point + 0.1)
                    if last_vector_size > 0:
                        plt.annotate('',
                                    xy=(last_point + 1, 0),
                                    xytext=(last_point + 1 + arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, 
                                                    shrinkA=0, shrinkB=0,
                                                    mutation_scale=arrow_head_size))
                    elif last_vector_size < 0:
                        plt.annotate('',
                                    xy=(last_point + 0.5, 0),
                                    xytext=(last_point + 0.5 + arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2,
                                                    shrinkA = 0, shrinkB=0,
                                                    mutation_scale=arrow_head_size))
                
                # Plot arrows or vectors for the single equilibrium case
                if len(real_EP) == 1:
                    single_point = float(real_EP[0])
                    vector_size = differential_equation_num(single_point - 0.1)  # Evaluate slightly to the left

                    # Plot an arrow to the left of the point
                    if vector_size < 0:
                        plt.annotate('',
                                    xy=(single_point - 0.5, 0),
                                    xytext=(single_point - 0.5 + arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, mutation_scale=arrow_head_size))
                    elif vector_size > 0:
                        plt.annotate('',
                                    xy=(single_point - 0.5, 0),
                                    xytext=(single_point - 0.5 - arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, mutation_scale=arrow_head_size))

                    vector_size_right = differential_equation_num(single_point + 0.1)  # Evaluate slightly to the right

                    # Plot an arrow to the right of the point
                    if vector_size_right > 0:
                        plt.annotate('',
                                    xy=(single_point + 0.5, 0),
                                    xytext=(single_point + 0.5 + arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, mutation_scale=arrow_head_size))
                    elif vector_size_right < 0:
                        plt.annotate('',
                                    xy=(single_point + 0.5, 0),
                                    xytext=(single_point + 0.5 - arrow_length, 0),
                                    arrowprops=dict(arrowstyle='->', color='green', lw=2, mutation_scale=arrow_head_size))

                # Ensure the plot covers a reasonable range for visualization
                if len(real_EP) == 1:
                    min_x = single_point - 0.5
                    max_x = single_point + 0.5
                    x = np.linspace(min_x, max_x, 1000)
                    x_prime = differential_equation_num(x)
                    plt.plot(x, x_prime, color=color, linewidth=2, zorder=1)
                    plt.hlines(0, min_x - 0.1, max_x + 0.1, color=color, linewidth=0.5, zorder=1)

                # Set axis limits for better visibility
                if len(real_EP) > 1:
                    plt.xlim(min_x - 1.5, max_x + 1.5)
                else:
                    plt.xlim(min_x, max_x)
                plt.ylim(min(x_prime) - 0.1, max(x_prime) + 0.1)

                # Remove the bezel (frame) around the plot
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)

                # Remove the ticks
                plt.yticks([])
                plt.xticks([])

                # plot the y-label
                plt.ylabel(axes_labels[1])

                # Position the x-axis label at the end of the line
                if len(real_EP) == 1:
                    plt.text(max_x + 0.15, 0, axes_labels[0], ha='center', va='center', fontsize=10)
                else:
                    plt.text(max_x + 1.55, 0, axes_labels[0], ha='center', va='center', fontsize=10)

                # Set the title
                if title:
                    plt.title(title, color = color, fontsize = 15)
                else:
                    plt.title(f'Linear Stability Analysis of: {differential_equation}', color = color, fontsize = 15)
                
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f'An error occured while plotting the linear stability analysis: {e}')
    except Exception as e:
        print(f'An error occurred while calculating and/or plotting the equilibrium points: {e}')
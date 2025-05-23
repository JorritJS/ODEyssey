# Import required modules
from typing import Callable, Tuple, List, Dict
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.markers import MarkerStyle
import inspect
from sympy import symbols, Eq, diff, solve, lambdify


def eulers_method(
        x_start: float,
        x_end: float,
        step_size: float,
        equation: Callable[[float], float],
        differential_equation: Callable[[float], float],
        plot_trajectory: bool = False,
        verbose: bool = False,
        dark_mode: bool = True,
        grid: bool = False,
        axes_labels: Tuple[str, str] = ("x", "y"),
        title: str = ""
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
        grid (bool, optional): boolean statement determining whether a grid is shown or not. Defaults to False.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        title (str, optional): If given, the plot title. Defaults to emtpy ("").

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

    # Preallocate arrays with the correct size (makes computation more efficient)
    solution_x_values = np.empty(steps + 1)
    solution_y_values = np.empty(steps + 1)

    # Initialize the first element since it will not be added in the for loop
    solution_x_values[0] = x_start
    solution_y_values[0] = equation(x_start) # Calculate with the equation since the value of y is function(x) where function(x) is our equation

    # Starting values of both x and y
    x, y = x_start, solution_y_values[0]

    # Define them locally to optimize the code
    ss = step_size # Local alias
    de = differential_equation # Local alias

    # Here we calculate our values by going over all the values in a step-wise manner
    if verbose:
        for i in range(1, steps + 1):
            x = x + ss # Update the x value; can be x += step_size but for clarity left like this
            y = y + ss * de(x) # Calculate our y value

            # Add them to the lists
            solution_x_values[i] = x
            solution_y_values[i] = y

            # Verbose
            print(f"x: {x:.4f}, y: {y:.6f}")
    else:
        for i in range(1, steps + 1):
            x = x + ss
            y = y + ss * de(x)
            solution_x_values[i] = x
            solution_y_values[i] = y

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
        
        plt.plot(solution_x_values, solution_y_values, marker='o', linestyle='-', color=color)
        plt.xlabel('x', fontsize=24)
        plt.ylabel('y', fontsize=24)

        # Title, if given then use that otherwise automatic one
        if title:
            plt.title(title, pad = 15, color = title_color, fontsize=28)
        else:
            title = f"Trajectory of {axes_labels[1]} as a Function of {axes_labels[0]} Using Euler\'s Method"
            plt.title(title, pad = 15, color = title_color, fontsize=28)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(grid)
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
        vector_colour: str = None,
        title: str = "",
        show: bool = True
    ) -> None:
    '''
    This function plots a vector field (2D) for two given equations.

    Args:
        equations Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Function representing the two equations.
        x_grid (Tuple[float, float]): Start and end values for the grid on the x-axis.
        y_grid (Tuple[float, float]): Start and end values for the grid on the y-axis.
        plot_points(int, optional): If given, the number of vectors plotted. Defaults to 20 points.
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        gradient (bool, optional): boolean statement determining whether a gradient of the vector size is shown or not. Defaults to True.
        grid (bool, optional): boolean statement determining whether a grid is shown or not. Defaults to False.
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (ggplot style). Defaults to True.
        vector_colour (str, optional): If given, use that colour. Defaults to None.
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
        x = np.linspace(x_grid[0], x_grid[1], plot_points)
        y = np.linspace(y_grid[0], y_grid[1], plot_points)
        X, Y = np.meshgrid(x, y, indexing="ij")  # Optimized indexing

        # Equations to plot
        u, v = vector_field_functions(X, Y)

        # Set plot style
        plt.style.use('dark_background' if dark_mode else 'default')
        title_color = 'white' if dark_mode else 'black'

        # To still be able to see the vectors with these exact conditions
        vector_colour = vector_colour or ('black' if dark_mode or gradient else 'white')

        # Gradient plot
        if gradient == True:
            magnitude = np.hypot(u, v)
            c = plt.imshow(magnitude, extent=(x_grid[0], x_grid[1], y_grid[0], y_grid[1]), interpolation='none', origin='lower')
            colorbar = plt.colorbar(c)
            colorbar.set_label('Relative length of vector', rotation=270, labelpad = 20, fontsize=14)

        # Plot
        plt.quiver(x, y, u, v, color = vector_colour)
        plt.xlabel(axes_labels[0], fontsize=24)
        plt.ylabel(axes_labels[1], fontsize=24)

        # Title, if given then use that otherwise automatic one
        if title:
            plt.title(title, pad = 15, color = title_color, fontsize=28)
        else:
            title = f"2D vector field superimposed on {axes_labels[0]}-{axes_labels[1]} state space"
            plt.title(title, pad = 15, color = title_color, fontsize=28)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(grid)
        
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

        ax.set_xlabel(axes_labels[0], fontsize=24)
        ax.set_ylabel(axes_labels[1], fontsize=24)
        ax.set_zlabel(axes_labels[2], fontsize=24)

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
        initial_state: list,
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
        step_size (float): size of the step we take for odeint function (comparable to Euler's step size).
        initial_state (list): where the trajectory starts.
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

        plot_trajectories(equations, 0, 40, 0.01, initial_state=initial_state, params=p)
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
        X_start (float): Starting value for the time series (time value, i.e. the starting value on the left side of the plot).
        x_stop (float): End value for the trajectory (time value, i.e. the end value on the right side of the plot).
        n_step (float): Number of steps we take for odeint function
        initial_state (List[float]): Initial state vector.
        interventions (List[Dict[str, float]], optional): List of interventions. Each intervention is a dictionary with 'time', 'type', and 'magnitude'.
        title (str, optional): If given, the plot title. Defaults to emtpy ("").
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        line_colors (List[str], optional): Colors for the lines. Defaults to Viridis colours.
        grid (bool, optional): Boolean statement determining whether a grid is shown. Defaults to True.
        dark_mode (bool, optional): Boolean statement determining whether the plot style is dark or not (default style). Defaults to True.

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
        title: str = None
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
            slope = (float(point))
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


def nullcline_plot(
    equations,
    vector_field = True,
    title: str = 'Nullclines of 2D differential system with equilibrium points',
    grid: bool = False,
    axes_labels: Tuple[str, str] = ('x','y'),
    dark_mode: bool = False
    ) -> None:
    """
    This function plots nullclines, equilibrium points and an optional vector field of a 2D system.

    Args:
        equations (Callable[[Tuple[float, float]], Tuple[float, float]]):
            Function representing the 2D system.
        vector_field (bool, optional): If True, plots the vector field of the system. Defaults to True.
        grid (bool, optional): Boolean statement determining whether a grid is shown or not. Defaults to False.
        axes_labels (Tuple[str, str], optional): Labels for the x and y axes. Defaults to ('x', 'y').
        title (str, optional): If given, the plot title. Defaults to empty ("").
        dark_mode (bool, optional): Boolean statement determining whether the plot style is dark or not. Defaults to True.

    Returns:
        None: Displays a plot with the nullclines and equilibrium points (and an optional vector field).
    
    Example:
        def equations(state):
            d_val, m_val = state
            dd_dt = 3*d_val - m_val*d_val - d_val**2
            dm_dt = 2*m_val - 0.5*m_val*d_val - m_val**2
            return dd_dt, dm_dt

        nullcline_plot(equations, title='Nullcline Plot')
    """
    try:
        d, m = symbols('d m') # I define it like this now, can also do global

        nullcline_one = solve(Eq(equations((d, m))[0], 0), d)
        nullcline_two = solve(Eq(equations((d, m))[1], 0), m)

        # Solve the system of equations
        EPs = solve([equations((d, m))[0], equations((d, m))[1]], (d, m))

        max_d = int(max(EP[0] for EP in EPs))
        max_m = int(max(EP[1] for EP in EPs))
        min_d = int(min(EP[0] for EP in EPs))
        min_m = int(min(EP[1] for EP in EPs))

        d_range = max_d - min_d
        m_range = max_m - min_m

        d_extension = d_range * 0.05
        m_extension = m_range * 0.05

        d_min_extended = min_d - d_extension
        d_max_extended = max_d + d_extension
        m_min_extended = min_m - m_extension
        m_max_extended = max_m + m_extension

        if dark_mode:
            plt.style.use('dark_background')
            color1 = 'white'
            color2 = 'black'
        else:
            plt.style.use('default')
            color1 = 'black'
            color2 = 'white'

        for nullcline_1 in nullcline_one:
            if nullcline_1.is_number:
                plt.axhline(y=nullcline_1, color='blue', xmin = 0, xmax =1)
            else:
                function = lambdify(m, nullcline_1)
                m_values = np.linspace(m_min_extended, m_max_extended, 100)
                plt.plot(m_values, function(m_values), color='blue')

        for nullcline in nullcline_two:
            if nullcline.is_number:
                plt.axvline(x=nullcline, color='red', ymin = 0, ymax = 1)
            else:
                function = lambdify(d, nullcline)
                d_values = np.linspace(d_min_extended, d_max_extended, 100)
                plt.plot(function(d_values), d_values, color='red')

        for EP in EPs:
            d_val, m_val = EP

            perturbation = [0.01, -0.01]
            stability_info = []

            for perturb in perturbation:
                # Check above
                above_dd = equations((d_val + perturb, m_val))[0]
                above_dm = equations((d_val, m_val + perturb))[1]
                # Below
                below_dd = equations((d_val - perturb, m_val))[0]
                below_dm = equations((d_val, m_val - perturb))[1]

                # Append
                stability_info.append(((above_dd, above_dm), (below_dd, below_dm)))

            above_dd, above_dm = stability_info[0][0]
            below_dd, below_dm = stability_info[1][0]

            # Check stability in the d direction
            stable_d = above_dd < 0 and below_dd > 0
            unstable_d = above_dd > 0 and below_dd < 0

            # Check stability in the m direction
            stable_m = above_dm < 0 and below_dm > 0
            unstable_m = above_dm > 0 and below_dm < 0

            # Determine overall stability
            if stable_d and stable_m:
                print(f'Stable equilibrium point: {EP}')
                plt.scatter(EP[1], EP[0], color = color1, zorder=2)
            elif unstable_d and unstable_m:
                print(f'Unstable equilibrium point: {EP}')
                plt.scatter(EP[1], EP[0], color = color2, edgecolors=color1, zorder=2)
            elif (stable_d and unstable_m) or (unstable_d and stable_m):
                print(f'Saddle point: {EP}')
                plt.scatter(EP[1], EP[0], color = color2, edgecolors=color1, zorder=2)
                plt.scatter(EP[1], EP[0], color = color1, marker = MarkerStyle('o', fillstyle='bottom'), zorder=2) # Fill it half
            else:
                print(f'Semi-stable equilibrium point: {EP}')
                plt.scatter(EP[1], EP[0], color = 'salmon', zorder=2)

        # Set the plot limits to the original equilibrium point ranges.
        plt.xlim(m_min_extended, m_max_extended)
        plt.ylim(d_min_extended, d_max_extended)
        
        # Optional vector field, taking the dimensions of the trajectory as the x_grid and y_grid. Show is set to False in plot_vector_field to prevent multiple outputs. 
        min_x, max_x = plt.xlim()
        min_y, max_y = plt.ylim()
        
        if vector_field:
            def vector_field_func(x, y):
                return equations((x, y))

            try:
                plot_vector_field(vector_field_func,
                                x_grid=(min_x, max_x),
                                y_grid=(min_y, max_y),
                                show = False,
                                gradient = False,
                                dark_mode = False,
                                grid = True,
                                vector_colour = 'green',
                                plot_points=10
                                )
            except Exception as e:
                print(f'Something went wrong while plotting the vector field: {e}')

        plt.xlabel(axes_labels[0])
        plt.ylabel(axes_labels[1])
        plt.grid(grid)
        plt.title(title, color = color1)
        plt.show()
    except Exception as e:
        print(f'An error occured while plotting the nullcline graph: {e}')


def bifurcation_diagram(
        differential_equation: Callable[[np.ndarray], np.ndarray],
        variable: str,
        bifurcation_symbol: str,
        bifurcation_start: float,
        bifurcation_end: float,
        n_step: float = 1000,
        axes_labels: Tuple[str, str] = ('x','y'),
        dark_mode: bool = True,
        grid: bool = False,
        title: str = ''
    ) -> None:
    '''
    This function creates a bifurcation diagram of a given differential equation by calculating its equilibrium points and
    analyzing their stability.

    Args:
        differential_equation (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): Function representing the differential equation.
        variable (str): The variable with respect to which the equilibrium points are calculated.
        bifurcation_symbol (str): The variable with respect to which the bifurcation is done.
        bifurcation_start (float): Starting value of the bifurcation symbol.
        bifurcation_end (float): End value of the bifurcation symbol.
        n_step (float): Number of values for the bifurcation we plot. Defaults to 1000.      
        axes_labels(Tuple[str, str], optional): If given, the axes labels for the x and y axes. Defaults to "x" and "y".
        dark_mode (bool, optional): boolean statement determining whether the plot style is dark or not (default style). Defaults to True.
        grid (bool, optional): Boolean statement determining whether a grid is shown or not. Defaults to False.
        title (str, optional): If given, the plot title. Defaults to None.
        
    Returns:
        plt: A plot with the bifurcation diagram.

    Example:
        X, a = symbols('X a')
        differential_eq = 2 * X * (1 - X / 5) * (X / a - 1)
        bifurcation_diagram(differential_eq, variable=X, bifurcation_symbol=a, bifurcation_start=0.1, bifurcation_end=20)
    '''
    try:
        # Get the values we check equilibrium points for
        bifurcation_values = np.linspace(bifurcation_start, bifurcation_end, n_step)
        if 0 in bifurcation_values:
            bifurcation_values = bifurcation_values[bifurcation_values != 0]
            print("bifurcation_start cannot be 0 as this will lead to division by 0, skipping 0 in bifurcation_values.")

        # Initialize equilibrium points list
        EP_stability_info = []

        # Analyze EPs and stability for each bifurcation value
        for bifurcation in bifurcation_values:
            # Substitute the bifurcation parameter into the equation
            substituted_equation = differential_equation.subs(bifurcation_symbol, bifurcation)
            differentiated_function = diff(substituted_equation, variable)
            equilibrium_points = solve(substituted_equation, variable)

            # Filter out the non-real equilibrium points
            real_points = [point.evalf() for point in equilibrium_points if point.is_real]

            # Analyze stability by first putting the equation into the right format
            differentiated_function_numeric = lambdify(variable, differentiated_function)

            for point in real_points:
                slope = differentiated_function_numeric(point)
                stability = "Stable" if slope < 0 else "Unstable" if slope > 0 else "Semi-stable"
                EP_stability_info.append((bifurcation, point, stability))

        # Get the data ready for plotting
        stable_points = [(bifurcation_point, stability_point) for bifurcation_point, stability_point, stability in EP_stability_info if stability == "Stable"]
        unstable_points = [(bifurcation_point, stability_point) for bifurcation_point, stability_point, stability in EP_stability_info if stability == "Unstable"]
        semi_stable_points = [(bifurcation_point, stability_point) for bifurcation_point, stability_point, stability in EP_stability_info if stability == "Semi-stable"]

        # Separate the coordinates
        stable_bifurcation, stable_equilibrium = zip(*stable_points) if stable_points else ([], [])
        unstable_bifurcation, unstable_equilibrium = zip(*unstable_points) if unstable_points else ([], [])
        semi_stable_bifurcation, semi_stable_equilibrium = zip(*semi_stable_points) if semi_stable_points else ([], [])

        if dark_mode:
            plt.style.use('dark_background')
            color = 'white'
        else:
            plt.style.use('default')
            color = 'black'

        # Plot the diagram
        plt.scatter(stable_bifurcation, stable_equilibrium, color='blue', label='Stable', s=10)
        plt.scatter(unstable_bifurcation, unstable_equilibrium, color='red', label='Unstable', s=10)
        plt.scatter(semi_stable_bifurcation, semi_stable_equilibrium, color='orange', label='Semi-stable', s=10)
        plt.xlabel(axes_labels[0], color=color)
        plt.ylabel(axes_labels[1], color=color)
        plt.grid(grid)
        plt.title(title, color=color)
        plt.show()
    except Exception as e:
        print(f'An error occured while plotting the bifurcation diagram; {e}')
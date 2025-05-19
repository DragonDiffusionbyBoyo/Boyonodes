import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import colorsys
import random
from PIL import Image, ImageFilter

def generate_vibrant_colormap(name="psychedelic", n_colors=256, randomize=False):
    """
    Generate a vibrant colormap with smooth transitions.
    
    Parameters:
    -----------
    name : str
        Name for the colormap
    n_colors : int
        Number of colors in the map
    randomize : bool
        Whether to add random variation to colors
        
    Returns:
    --------
    matplotlib.colors.ListedColormap
        A colormap for visualization
    """
    colors = []
    
    if randomize:
        golden_ratio = 0.618033988749895
        hue = random.random()
        for i in range(n_colors):
            hue = (hue + golden_ratio) % 1.0
            saturation = 0.7 + random.random() * 0.3
            value = 0.8 + random.random() * 0.2
            colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    else:
        for i in range(n_colors):
            r = 0.5 + 0.5 * np.sin(0.1 * i + 0)
            g = 0.5 + 0.5 * np.sin(0.1 * i + 2)
            b = 0.5 + 0.5 * np.sin(0.1 * i + 4)
            colors.append((r, g, b))
    
    colors[-1] = (0, 0, 0)
    
    return ListedColormap(colors, name=name)

def smooth_coloring(iterations, z, max_iter, smooth_method="log"):
    """
    Apply smooth coloring to the fractal.
    
    Parameters:
    -----------
    iterations : np.ndarray
        Array of iteration counts
    z : np.ndarray
        Final complex value for each point
    max_iter : int
        Maximum number of iterations
    smooth_method : str
        Smoothing method to use: 'log', 'sqrt', or 'linear'
        
    Returns:
    --------
    np.ndarray
        Smoothed color values
    """
    mask = iterations < max_iter
    smooth = np.copy(iterations).astype(float)
    
    if mask.any():
        if smooth_method == "log":
            log_zn = np.zeros_like(z, dtype=float)
            log_zn[mask] = np.log(np.abs(z[mask])) / 2
            nu = np.zeros_like(z, dtype=float)
            nu[mask] = np.log(log_zn[mask]) / np.log(2)
            smooth[mask] = iterations[mask] + 1 - nu[mask]
        elif smooth_method == "sqrt":
            smooth[mask] = iterations[mask] + 1 - np.log(np.log(np.abs(z[mask]))) / np.log(2)
        else:
            abs_z = np.abs(z)
            smooth[mask] = iterations[mask] + 1 - (np.log(np.log(abs_z[mask])) / np.log(2))
    
    return smooth

def compute_mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max, julia_c=None, julia_power=2, smooth=True):
    """
    Compute the Mandelbrot set (or Julia set) within the specified region.
    
    Parameters:
    -----------
    h, w : int
        Height and width of the output image in pixels.
    max_iter : int
        Maximum number of iterations to determine if a point is in the set.
    x_min, x_max, y_min, y_max : float
        Boundaries of the region in the complex plane.
    julia_c : complex, optional
        If not None, computes a Julia set for the given c parameter.
    julia_power : int, default=2
        Power to raise z to in the iteration formula (z = z^power + c).
    smooth : bool, default=True
        Whether to use smooth coloring.
        
    Returns:
    --------
    np.ndarray
        2D array with color values.
    """
    y, x = np.ogrid[y_max:y_min:h*1j, x_min:x_max:w*1j]
    
    if julia_c is None:
        c = x + y*1j
        z = np.zeros_like(c)
    else:
        z = x + y*1j
        c = np.ones_like(z) * julia_c
    
    iterations = np.zeros(z.shape, dtype=int)
    mask = np.ones(z.shape, dtype=bool)
    
    for i in range(max_iter):
        mask[np.abs(z) > 2.0] = False
        iterations[~mask & (iterations == 0)] = i
        z[mask] = z[mask]**julia_power + c[mask]
        if not np.any(mask):
            break
    
    iterations[mask] = max_iter
    
    if smooth:
        result = smooth_coloring(iterations, z, max_iter, smooth_method="log")
        if np.max(result) > np.min(result):
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        result = result * max_iter
    else:
        result = iterations
    
    return result

def convolve2d(image, kernel):
    """
    Perform 2D convolution using NumPy.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image array
    kernel : np.ndarray
        Convolution kernel
        
    Returns:
    --------
    np.ndarray
        Convolved image
    """
    kh, kw = kernel.shape
    h, w = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    
    return output

def apply_effects(data, effects=None):
    """
    Apply visual effects to the fractal data.
    
    Parameters:
    -----------
    data : np.ndarray
        The fractal data
    effects : dict, optional
        Dictionary of effects to apply
        
    Returns:
    --------
    np.ndarray
        Modified fractal data
    """
    if effects is None:
        return data
    
    result = data.copy()
    
    if 'invert' in effects and effects['invert']:
        max_val = np.max(result)
        result = max_val - result
    
    if 'sobel' in effects and effects['sobel']:
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_x = convolve2d(result, sobel_x)
        grad_y = convolve2d(result, sobel_y)
        sobel = np.sqrt(grad_x**2 + grad_y**2)
        result = 0.5 * result + 0.5 * sobel
    
    if 'emboss' in effects and effects['emboss']:
        emboss_kernel = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
        emboss = convolve2d(result, emboss_kernel)
        result = 0.5 * result + 0.5 * emboss
    
    if 'gaussian' in effects:
        sigma = effects.get('gaussian', 1.0)
        result_uint8 = (result / np.max(result) * 255).astype(np.uint8)
        img = Image.fromarray(result_uint8, mode='L')
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        result = np.array(img_blurred).astype(np.float64) / 255 * np.max(result)
    
    if 'contrast' in effects:
        contrast = effects.get('contrast', 1.0)
        mean = np.mean(result)
        result = (result - mean) * contrast + mean
    
    if 'color_cycle' in effects:
        cycle_factor = effects.get('color_cycle', 3.0)
        result = result * cycle_factor % np.max(result)
    
    if 'bands' in effects and effects['bands']:
        band_factor = effects.get('band_factor', 3.0)
        result = np.sin(result * band_factor) * np.max(result) / 2 + result / 2
    
    if np.max(result) > np.min(result):
        result = (result - np.min(result)) / (np.max(result) - np.min(result)) * np.max(data)
    
    return result

def plot_mandelbrot(fractal_data, x_min, x_max, y_min, y_max, 
                    colormap='viridis', title="Mandelbrot Set", 
                    invert_colors=False, gradient_repeat=1,
                    hide_axes=False):
    """
    Plot the fractal with a custom colormap.
    
    Parameters:
    -----------
    fractal_data : np.ndarray
        2D array with fractal data.
    x_min, x_max, y_min, y_max : float
        Boundaries of the region in the complex plane.
    colormap : str or matplotlib.colors.Colormap
        Colormap to use for the plot.
    title : str
        Title of the plot.
    invert_colors : bool
        Whether to invert the colormap.
    gradient_repeat : int
        Number of times to repeat the color gradient.
    hide_axes : bool
        Whether to hide the axes.
    """
    plt.figure(figsize=(12, 10))
    
    if gradient_repeat > 1:
        fractal_data = fractal_data * gradient_repeat % np.max(fractal_data)
    
    if invert_colors:
        fractal_data = np.max(fractal_data) - fractal_data
    
    if isinstance(colormap, str) and colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap
    
    max_val = np.max(fractal_data)
    if np.any(fractal_data == max_val):
        masked_data = np.ma.masked_where(fractal_data == max_val, fractal_data)
        img = plt.imshow(masked_data, cmap=cmap, extent=[x_min, x_max, y_min, y_max])
        plt.imshow(np.ones_like(fractal_data), 
                 cmap=ListedColormap(['black']),
                 extent=[x_min, x_max, y_min, y_max],
                 alpha=np.where(fractal_data == max_val, 1, 0))
    else:
        img = plt.imshow(fractal_data, cmap=cmap, extent=[x_min, x_max, y_min, y_max])
    
    plt.title(title, fontsize=16)
    
    if hide_axes:
        plt.axis('off')
    else:
        plt.xlabel('Re(c)')
        plt.ylabel('Im(c)')
        plt.colorbar(img, label='Value')
    
    plt.tight_layout()
    return plt.gcf()

def create_color_schemes():
    """
    Create a dictionary of custom color schemes.
    
    Returns:
    --------
    dict
        Dictionary of custom colormaps
    """
    schemes = {}
    
    colors = []
    for i in range(256):
        r = 0.5 + 0.5 * np.sin(0.1 * i + 0)
        g = 0.5 + 0.5 * np.sin(0.1 * i + 2.094)
        b = 0.5 + 0.5 * np.sin(0.1 * i + 4.188)
        colors.append((r, g, b))
    schemes['rainbow_psychedelic'] = ListedColormap(colors)
    
    colors = []
    for i in range(256):
        r = min(1.0, i / 128)
        g = max(0.0, min(1.0, (i - 64) / 64))
        b = max(0.0, min(1.0, (i - 160) / 32))
        colors.append((r, g, b))
    schemes['fire'] = ListedColormap(colors)
    
    colors = []
    for i in range(256):
        r = max(0.0, min(1.0, (i - 196) / 64))
        g = max(0.0, min(1.0, (i - 128) / 64))
        b = min(1.0, i / 128)
        colors.append((r, g, b))
    schemes['electric_blue'] = ListedColormap(colors)
    
    colors = []
    for i in range(256):
        r = 0
        g = i / 510
        b = min(1.0, i / 255)
        colors.append((r, g, b))
    schemes['deep_sea'] = ListedColormap(colors)
    
    colors = []
    for i in range(256):
        phase = i / 255 * 6 * np.pi
        r = max(0, min(1, np.sin(phase) + 0.5))
        g = max(0, min(1, np.sin(phase + 2) + 0.5))
        b = max(0, min(1, np.sin(phase + 4) + 0.5))
        colors.append((r, g, b))
    schemes['neon'] = ListedColormap(colors)
    
    colors = []
    golden_ratio = 0.618033988749895
    h = 0.5
    for i in range(256):
        h = (h + golden_ratio) % 1.0
        s = 0.8 + 0.2 * np.sin(i * 0.1)
        v = 0.9 + 0.1 * np.cos(i * 0.05)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append(rgb)
    schemes['ultra_vibrant'] = ListedColormap(colors)
    
    return schemes

def explore_mandelbrot(h=800, w=1000, max_iter=200, 
                       x_min=-2.5, x_max=1.0, y_min=-1.2, y_max=1.2,
                       colormap='ultra_vibrant', title="Mandelbrot Set",
                       julia_c=None, julia_power=2, hide_axes=False,
                       gradient_repeat=1, smooth=True, effects=None,
                       invert_colors=False, random_colors=False):
    """
    Compute and display the Mandelbrot set with enhanced visual options.
    """
    fractal_type = "Julia Set" if julia_c is not None else "Mandelbrot Set"
    print(f"Computing {fractal_type} with resolution {w}x{h} and {max_iter} max iterations...")
    print(f"Region: [{x_min}, {x_max}] × [{y_min}, {y_max}]")
    
    color_schemes = create_color_schemes()
    
    if random_colors:
        cmap = generate_vibrant_colormap(randomize=True)
    elif colormap in color_schemes:
        cmap = color_schemes[colormap]
    else:
        cmap = colormap
    
    fractal_data = compute_mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max, 
                                     julia_c=julia_c, julia_power=julia_power, smooth=smooth)
    
    if effects:
        fractal_data = apply_effects(fractal_data, effects)
    
    fig = plot_mandelbrot(fractal_data, x_min, x_max, y_min, y_max, 
                         colormap=cmap, title=title, invert_colors=invert_colors,
                         gradient_repeat=gradient_repeat, hide_axes=hide_axes)
    
    if not smooth:
        in_set = np.sum(fractal_data == max_iter)
        print(f"Points in set: {in_set} ({in_set/(h*w)*100:.2f}%)")
    
    plt.show()
    return fractal_data, fig

regions = {
    "full_view": {"x_min": -2.5, "x_max": 1.0, "y_min": -1.2, "y_max": 1.2},
    "seahorse_valley": {"x_min": -0.75, "x_max": -0.74, "y_min": 0.09, "y_max": 0.11},
    "elephant_valley": {"x_min": 0.25, "x_max": 0.30, "y_min": 0.0, "y_max": 0.05},
    "spiral": {"x_min": -0.745, "x_max": -0.743, "y_min": 0.126, "y_max": 0.128},
    "mini_mandelbrot": {"x_min": -1.8, "x_max": -1.7, "y_min": -0.05, "y_max": 0.05},
    "tendrils": {"x_min": -0.22, "x_max": -0.21, "y_min": 0.64, "y_max": 0.65},
    "deep_zoom": {"x_min": -0.7435, "x_max": -0.7434, "y_min": 0.1314, "y_max": 0.1315},
}

julia_sets = {
    "dendrite": -0.123 + 0.745j,
    "douady_rabbit": -0.123 + 0.745j,
    "san_marco": -0.75,
    "siegel_disk": -0.391 - 0.587j,
    "galaxy": 0.285 + 0.01j,
    "lakes": -0.70176 - 0.3842j,
}

if __name__ == "__main__":
    fractal_data, fig = explore_mandelbrot(
        h=800, 
        w=1000,
        max_iter=300,
        colormap='ultra_vibrant',
        gradient_repeat=3,
        smooth=True,
        effects={
            'color_cycle': 5,
            'bands': True,
            'contrast': 1.2,
        },
        title="Ultra Vibrant Mandelbrot"
    )
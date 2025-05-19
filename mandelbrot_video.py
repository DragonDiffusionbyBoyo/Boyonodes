import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import colorsys
import random
import os
import shutil
import argparse
from tqdm import tqdm
import subprocess
from PIL import Image
plt.switch_backend('Agg')
from mandelbrot_generator import (
    compute_mandelbrot, apply_effects, create_color_schemes, regions, julia_sets,
    generate_vibrant_colormap
)

FPS = 30
DURATION = 6
TOTAL_FRAMES = FPS * DURATION

RESOLUTIONS = {
    "square": (512, 512),
    "widescreen": (720, 480),
    "vertical": (480, 836)
}

def interpolate_linear(start_val, end_val, t):
    return start_val * (1 - t) + end_val * t

def interpolate_log(start_val, end_val, t):
    if start_val <= 0 or end_val <= 0:
        return interpolate_linear(start_val, end_val, t)
    log_start = np.log(start_val)
    log_end = np.log(end_val)
    log_result = interpolate_linear(log_start, log_end, t)
    return np.exp(log_result)

def interpolate_region(start_region, end_region, t):
    region = {}
    if (end_region["x_max"] - end_region["x_min"]) < (start_region["x_max"] - start_region["x_min"]):
        width_start = start_region["x_max"] - start_region["x_min"]
        width_end = end_region["x_max"] - end_region["x_min"]
        height_start = start_region["y_max"] - start_region["y_min"]
        height_end = end_region["y_max"] - end_region["y_min"]
        new_width = interpolate_log(width_start, width_end, t)
        new_height = interpolate_log(height_start, height_end, t)
        center_x_start = (start_region["x_min"] + start_region["x_max"]) / 2
        center_x_end = (end_region["x_min"] + end_region["x_max"]) / 2
        center_y_start = (start_region["y_min"] + start_region["y_max"]) / 2
        center_y_end = (end_region["y_min"] + end_region["y_max"]) / 2
        center_x = interpolate_linear(center_x_start, center_x_end, t)
        center_y = interpolate_linear(center_y_start, center_y_end, t)
        region["x_min"] = center_x - new_width / 2
        region["x_max"] = center_x + new_width / 2
        region["y_min"] = center_y - new_height / 2
        region["y_max"] = center_y + new_height / 2
    else:
        for key in ["x_min", "x_max", "y_min", "y_max"]:
            region[key] = interpolate_linear(start_region[key], end_region[key], t)
    return region

def interpolate_effects(start_effects, end_effects, t):
    effects = {}
    all_keys = set(list(start_effects.keys()) + list(end_effects.keys()))
    for key in all_keys:
        if key in ["invert", "sobel", "emboss", "bands"]:
            if t < 0.5:
                if key in start_effects:
                    effects[key] = start_effects[key]
            else:
                if key in end_effects:
                    effects[key] = end_effects[key]
        elif key in ["gaussian", "contrast", "color_cycle", "band_factor"]:
            start_val = start_effects.get(key, 0 if key != "contrast" else 1.0)
            end_val = end_effects.get(key, 0 if key != "contrast" else 1.0)
            effects[key] = interpolate_linear(start_val, end_val, t)
    return effects

def interpolate_colormap(start_cmap, end_cmap, t, n_colors=256):
    color_schemes = create_color_schemes()
    if isinstance(start_cmap, str):
        if start_cmap in color_schemes:
            start_cmap = color_schemes[start_cmap]
        elif start_cmap in plt.colormaps():
            start_cmap = plt.get_cmap(start_cmap)
        else:
            start_cmap = plt.get_cmap('viridis')
    if isinstance(end_cmap, str):
        if end_cmap in color_schemes:
            end_cmap = color_schemes[end_cmap]
        elif end_cmap in plt.colormaps():
            end_cmap = plt.get_cmap(end_cmap)
        else:
            end_cmap = plt.get_cmap('viridis')
    start_colors = start_cmap(np.linspace(0, 1, n_colors))
    end_colors = end_cmap(np.linspace(0, 1, n_colors))
    mixed_colors = []
    for i in range(n_colors):
        r = interpolate_linear(start_colors[i][0], end_colors[i][0], t)
        g = interpolate_linear(start_colors[i][1], end_colors[i][1], t)
        b = interpolate_linear(start_colors[i][2], end_colors[i][2], t)
        a = interpolate_linear(start_colors[i][3], end_colors[i][3], t)
        mixed_colors.append((r, g, b, a))
    return ListedColormap(mixed_colors)

def get_frame_settings(keyframes, frame_idx, total_frames):
    position = frame_idx / (total_frames - 1)
    prev_keyframe = keyframes[0]
    next_keyframe = keyframes[-1]
    prev_pos = 0
    next_pos = 1
    for i, kf in enumerate(keyframes):
        kf_pos = kf.get("position", i / (len(keyframes) - 1))
        if kf_pos <= position and kf_pos > prev_pos:
            prev_keyframe = kf
            prev_pos = kf_pos
        if kf_pos > position and kf_pos < next_pos:
            next_keyframe = kf
            next_pos = kf_pos
    if position == prev_pos:
        return prev_keyframe.copy()
    t = (position - prev_pos) / (next_pos - prev_pos) if next_pos > prev_pos else 0
    t = 0.5 - 0.5 * np.cos(t * np.pi)
    settings = {}
    settings["region"] = interpolate_region(
        {k: prev_keyframe["region"][k] for k in ["x_min", "x_max", "y_min", "y_max"]},
        {k: next_keyframe["region"][k] for k in ["x_min", "x_max", "y_min", "y_max"]},
        t
    )
    if "julia_c" in prev_keyframe or "julia_c" in next_keyframe:
        prev_c = prev_keyframe.get("julia_c")
        next_c = next_keyframe.get("julia_c")
        if prev_c is not None and next_c is not None:
            real = interpolate_linear(prev_c.real, next_c.real, t)
            imag = interpolate_linear(prev_c.imag, next_c.imag, t)
            settings["julia_c"] = complex(real, imag)
        else:
            if t < 0.5:
                settings["julia_c"] = prev_c
            else:
                settings["julia_c"] = next_c
    for key in ["max_iter", "gradient_repeat", "julia_power"]:
        if key in prev_keyframe or key in next_keyframe:
            prev_val = prev_keyframe.get(key, 200 if key == "max_iter" else 1)
            next_val = next_keyframe.get(key, 200 if key == "max_iter" else 1)
            settings[key] = int(round(interpolate_linear(prev_val, next_val, t)))
    for key in ["smooth", "invert_colors", "hide_axes"]:
        if key in prev_keyframe or key in next_keyframe:
            if t < 0.5:
                settings[key] = prev_keyframe.get(key, False)
            else:
                settings[key] = next_keyframe.get(key, False)
    if "effects" in prev_keyframe or "effects" in next_keyframe:
        prev_effects = prev_keyframe.get("effects", {})
        next_effects = next_keyframe.get("effects", {})
        settings["effects"] = interpolate_effects(prev_effects, next_effects, t)
    if "julia_c" in settings:
        settings["title"] = f"Julia Set (c={settings['julia_c']})"
    else:
        settings["title"] = "Mandelbrot Set"
    if "colormap" in prev_keyframe or "colormap" in next_keyframe:
        prev_cmap = prev_keyframe.get("colormap", "viridis")
        next_cmap = next_keyframe.get("colormap", "viridis")
        if prev_cmap == next_cmap:
            settings["colormap"] = prev_cmap
        else:
            settings["colormap"] = interpolate_colormap(prev_cmap, next_cmap, t)
    for key, value in prev_keyframe.items():
        if key not in settings and key != "position":
            settings[key] = value
    return settings

def render_frame(settings, width, height, frame_idx=None, grayscale=False):
    region = settings["region"]
    x_min, x_max = region["x_min"], region["x_max"]
    y_min, y_max = region["y_min"], region["y_max"]
    max_iter = settings.get("max_iter", 200)
    smooth = settings.get("smooth", True)
    julia_c = settings.get("julia_c", None)
    julia_power = settings.get("julia_power", 2)
    colormap = settings.get("colormap", "viridis")
    title = settings.get("title", "Mandelbrot Set" if julia_c is None else f"Julia Set (c={julia_c})")
    gradient_repeat = settings.get("gradient_repeat", 1)
    effects = settings.get("effects", {})
    invert_colors = settings.get("invert_colors", False)
    hide_axes = settings.get("hide_axes", True)
    if frame_idx is not None:
        title = f"{title} - Frame {frame_idx}"
    fractal_data = compute_mandelbrot(
        height, width, max_iter, x_min, x_max, y_min, y_max,
        julia_c=julia_c, julia_power=julia_power, smooth=smooth
    )
    fractal_data = fractal_data.astype(np.float64)
    if effects:
        fractal_data = apply_effects(fractal_data, effects)
        fractal_data = fractal_data.astype(np.float64)
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_position([0, 0, 1, 1])
    if gradient_repeat > 1:
        fractal_data = fractal_data * gradient_repeat % np.max(fractal_data)
        fractal_data = fractal_data.astype(np.float64)
    if invert_colors:
        fractal_data = np.max(fractal_data) - fractal_data
        fractal_data = fractal_data.astype(np.float64)
    color_schemes = create_color_schemes()
    if isinstance(colormap, str) and colormap in color_schemes:
        cmap = color_schemes[colormap]
    elif isinstance(colormap, str) and colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = colormap
    max_val = np.max(fractal_data)
    if np.any(fractal_data == max_val):
        masked_data = np.ma.masked_where(fractal_data == max_val, fractal_data)
        alpha = np.where(fractal_data == max_val, 1.0, 0.0).astype(np.float64)
        img = ax.imshow(masked_data, cmap=cmap, extent=[x_min, x_max, y_min, y_max])
        ax.imshow(np.ones_like(fractal_data),
                 cmap=ListedColormap(['black']),
                 extent=[x_min, x_max, y_min, y_max],
                 alpha=alpha)
    else:
        img = ax.imshow(fractal_data, cmap=cmap, extent=[x_min, x_max, y_min, y_max])
    if hide_axes:
        ax.axis('off')
    else:
        ax.set_xlabel('Re(c)')
        ax.set_ylabel('Im(c)')
    temp_file = "temp_frame.png"
    fig.savefig(temp_file, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    img = Image.open(temp_file)
    if grayscale:
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    os.remove(temp_file)
    return img

def create_default_keyframes():
    random.seed()
    keyframes = []
    available_regions = list(regions.keys())
    available_julia = list(julia_sets.keys())
    color_schemes = create_color_schemes()
    available_colormaps = list(color_schemes.keys())
    def random_region():
        if random.random() < 0.5:
            return regions[random.choice(available_regions)].copy()
        else:
            center_x = random.uniform(-2.0, 1.0)
            center_y = random.uniform(-1.5, 1.5)
            scale = random.uniform(0.01, 0.5)
            return {
                "x_min": center_x - scale,
                "x_max": center_x + scale,
                "y_min": center_y - scale,
                "y_max": center_y + scale
            }
    def random_effects():
        possible_effects = ["color_cycle", "contrast", "bands", "gaussian"]
        effects = {}
        for effect in possible_effects:
            if random.random() < 0.7:
                if effect == "color_cycle":
                    effects[effect] = random.uniform(2.0, 8.0)
                elif effect == "contrast":
                    effects[effect] = random.uniform(1.0, 2.0)
                elif effect == "bands":
                    effects[effect] = True
                    effects["band_factor"] = random.uniform(2.0, 5.0)
                elif effect == "gaussian":
                    effects[effect] = random.uniform(0.3, 1.0)
        return effects
    keyframes.append({
        "position": 0.0,
        "region": regions["full_view"].copy(),
        "max_iter": random.randint(150, 300),
        "colormap": random.choice(available_colormaps + [generate_vibrant_colormap(randomize=True)]),
        "gradient_repeat": random.randint(1, 4),
        "effects": random_effects(),
        "smooth": True,
        "hide_axes": True
    })
    for pos in [0.2, 0.4, 0.6, 0.8]:
        is_julia = random.random() < 0.3
        if is_julia:
            julia_c = julia_sets[random.choice(available_julia)]
            region = {
                "x_min": random.uniform(-1.5, -0.5),
                "x_max": random.uniform(0.5, 1.5),
                "y_min": random.uniform(-1.5, -0.5),
                "y_max": random.uniform(0.5, 1.5)
            }
        else:
            julia_c = None
            region = random_region()
        keyframes.append({
            "position": pos,
            "region": region,
            "max_iter": random.randint(200, 400),
            "colormap": random.choice(available_colormaps + [generate_vibrant_colormap(randomize=True)]),
            "gradient_repeat": random.randint(1, 6),
            "effects": random_effects(),
            "smooth": True,
            "hide_axes": True,
            "julia_c": julia_c
        })
    keyframes.append({
        "position": 1.0,
        "region": random_region() if random.random() < 0.5 else regions["full_view"].copy(),
        "max_iter": random.randint(150, 300),
        "colormap": random.choice(available_colormaps + [generate_vibrant_colormap(randomize=True)]),
        "gradient_repeat": random.randint(1, 4),
        "effects": random_effects(),
        "smooth": True,
        "hide_axes": True
    })
    return keyframes

def generate_video(width, height, output_file="mandelbrot_video.mp4", keyframes=None, fps=30, duration=6, grayscale=False):
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    if keyframes is None:
        keyframes = create_default_keyframes()
    total_frames = fps * duration
    print(f"Generating {total_frames} frames for a {duration}s video at {fps} FPS...")
    for frame_idx in tqdm(range(total_frames)):
        settings = get_frame_settings(keyframes, frame_idx, total_frames)
        img = render_frame(settings, width, height, grayscale=grayscale)
        frame_file = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        img.save(frame_file)
    print("Compiling video with ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_file
    ]
    subprocess.run(cmd, check=True)
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print(f"Video saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate a dynamic Mandelbrot set video")
    parser.add_argument("--resolution", choices=["square", "widescreen", "vertical"], default="square")
    parser.add_argument("--output", default="mandelbrot_video.mp4")
    parser.add_argument("--output_frames", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    width, height = RESOLUTIONS[args.resolution]
    print(f"Generating Mandelbrot frames at resolution {width}x{height}")

    keyframes = create_default_keyframes()
    temp_dir = os.path.join(os.path.dirname(args.output), "mandelbrot_frames")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    total_frames = FPS * DURATION
    for frame_idx in tqdm(range(total_frames)):
        settings = get_frame_settings(keyframes, frame_idx, total_frames)
        img = render_frame(settings, width, height, grayscale=args.grayscale)
        frame_file = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        img.save(frame_file)

    if args.output_frames:
        print(f"Frames saved to: {temp_dir}")
        return

    print("Compiling video with ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(temp_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        args.output
    ]
    subprocess.run(cmd, check=True)
    print(f"Video saved to: {args.output}")

if __name__ == "__main__":
    main()
import argparse
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="Convert ISO (Super) Duplex 120 stereos to gifs.")
parser.add_argument(
    '-p', '--path',
    type=lambda p: Path(p).absolute(),
    help='Path to directory with images to convert. *.jpg only. If not specified - script directory.',
    default=Path(Path(__file__).absolute().parent))
args = parser.parse_args()


def search_borders(image):
    """

    :param image: half of initial stereo pair.
    :return: top left corner x and y, width and height.
    """
    # Convert to grayscale OpenCV image.
    mask = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Speed up process by resize image to be smaller than 1000px in any direction.
    width, height = image.size
    scale = ceil(max(width, height) / 1000)
    mask = cv2.resize(mask, (int(width / scale), int(height / scale)))

    # Black pads to prevent any edge effects in follow operations.
    pad = 50
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value=0)

    # Apply morphology to remove small features. Blur...kind of.
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Separate horizontal and vertical lines to filter out spots.
    kernel = np.ones((12, 3), np.uint8)
    vert = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 12), np.uint8)
    horiz = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # Combine
    rect = cv2.add(horiz, vert)

    # Left frame has a key feature about 3 mm height, that must not be selected.
    # Morphology with large enough kernel help.
    kernel = np.ones((121, 121), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)
    rect = cv2.morphologyEx(rect, cv2.MORPH_DILATE, kernel)

    # Frame border typically rough, so trim it slightly inward.
    # Give better result than trimming or scaling bounding box.
    kernel = np.ones((17, 17), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)

    # Convert to binary.
    rect = cv2.threshold(rect, 32, 255, cv2.THRESH_BINARY)[1]

    # Crop pads added in the beginning.
    rect = rect[pad:pad + width, pad:pad + height]

    # Find rectangles and sort by area, largest first.
    contours, _ = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Take up to 5 largest contours, find its smallest bounding.
    x, y, w, h = cv2.boundingRect(np.concatenate(contours[0:min(5, len(contours))]))

    # Unscale results.
    return x * scale, y * scale, w * scale, h * scale


def split(image):
    """
    Split raw image in half.

    :param image: raw image with two image areas and mostly black background.
    :return: left and right part.
    """
    width, height = image.size
    cut_point = int(width / 2)
    return image.crop((0, 0, cut_point, height)), image.crop((cut_point, 0, width, height))


def fps_as_duration(fps):
    """
    Convert frames per second to frame duration.

    :param fps: frames per second. Only ints supported.
    :return: frame duration in milliseconds.
    """
    return int(1 / int(fps) * 1000)


# Size for result GIF image. ISO (Super) Duplex 120 has 23.5x25mm frame.
# 1000 height is typical maximum to autoplay GIF in social media.
output_size = (940, 1000)

# Get all jpg files in directory.
for path in Path(args.path).glob('*.jpg'):
    print('Process: %s ...' % path)
    with Image.open(path).convert(mode='RGB') as raw:
        # Part image in two halves.
        left, right = split(raw)
        # Search borders on left image.
        left_x, left_y, left_width, left_height = search_borders(left)
        # Crop and resize for output.
        left = left.crop(
            [left_x, left_y, left_x + left_width, left_y + left_height]) \
            .resize(output_size, resample=Image.Resampling.LANCZOS)
        # Search borders on right image. Only top and left will be used.
        right_x, right_y, _, _ = search_borders(right)
        # Crop right image to be exactly the same size as left image, and resize for output.
        right = right.crop(
            [right_x, right_y, right_x + left_width, right_y + left_height]) \
            .resize(output_size, resample=Image.Resampling.LANCZOS)
    # Save as GIF
    left.save(
        format='GIF',
        fp=Path(path).with_suffix('.gif'),  # Use original file name, with new extension.
        save_all=True,  # Save appended images.
        append_images=[right],  # Add right image as second frame.
        optimize=True,  # Optimize pallet if it can be done.
        disposal=1,  # Use first frame as background. This prevent ripping on loop.
        duration=fps_as_duration(6),
        loop=0)  # Infinite loop.
    # Tel user that we done with this image.
    print('Process: %s - OK' % path)

print('Ready!')

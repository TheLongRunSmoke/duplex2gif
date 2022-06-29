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


def search_horizontal_border_size(image_data, height_range, width, threshold):
    """
    Search border edge on top or bottom side of image.

    :param image_data: image as PIL.PixelAccess.
    :param height_range: range where to search for border.
    :param width: image_data width.
    :param threshold: 0.0..1.0, percent of white pixels in row to determinate as image area.
    :return: Row number where border end.
    """
    # PixelAccess 8-bit, but mask 1-bit. All pixels 0 or 255.
    max_value = width * 255
    # Iterate through range, until find row with more white pixels than threshold allow.
    for y in height_range:
        value = 0
        for x in range(width):
            value += image_data[x, y]
            if (value / max_value) > threshold:
                return y
    raise ValueError("Can't found border in given range.")


def search_vertical_border_size(image_data, width_range, height, threshold):
    """
    Search border edge on left or right side of image.

    :param image_data: image as PIL.PixelAccess.
    :param width_range: range where to search for border.
    :param height: image_data height.
    :param threshold: 0.0..1.0, percent of white pixels in column to determinate as image area.
    :return: Column number where border end.
    """
    # PixelAccess 8-bit, but mask 1-bit. All pixels 0 or 255.
    max_value = height * 255
    # Iterate through range, until find column with more white pixels than threshold allow.
    for x in width_range:
        value = 0
        for y in range(height):
            value += image_data[x, y]
            if (value / max_value) > threshold:
                return x
    raise ValueError("Can't found border in given range.")


def search_borders(image):
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    width, height = image.size
    scale = ceil(max(width, height) / 1000)
    mask = cv2.resize(mask, (int(width / scale), int(height / scale)))
    mask = 255 - mask

    # mask = cv2.mask.filter()

    # do adaptive threshold on gray image
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mask = clahe.apply(mask)

    # mask = cv2.equalizeHist(mask)
    # mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 1)
    thresh = 255 - mask

    # apply morphology
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # separate horizontal and vertical lines to filter out spots outside the rectangle
    kernel = np.ones((9, 3), np.uint8)
    vert = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 9), np.uint8)
    horiz = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # combine
    rect = cv2.add(horiz, vert)

    # thin
    kernel = np.ones((7, 7), np.uint8)
    rect = cv2.morphologyEx(rect, cv2.MORPH_ERODE, kernel)
    # mask = cv2.erode(mask, kernel, iterations=10)
    # mask = cv2.dilate(mask, kernel, iterations=10)

    # get largest contour
    contours, _ = cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda cnt: cv2.contourArea(cnt) > 200, contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x, y, w, h = cv2.boundingRect(np.concatenate(contours[0:min(3, len(contours))]))
    x *= scale
    y *= scale
    w *= scale
    h *= scale
    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 10)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x *= scale
        y *= scale
        w *= scale
        h *= scale
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36, 255, 12), 3)

    # get rotated rectangle from contour
    # rot_rect = cv2.minAreaRect(big_contour)
    # box = cv2.boxPoints(rot_rect)
    # box = np.int0(box)

    # draw rotated rectangle on copy of img
    # cv_image = cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 2)

    # # get rotated rectangle from contour
    # rot_rect = cv2.minAreaRect(big_contour)
    # box = cv2.boxPoints(rot_rect)
    # box = np.int0(box)
    # # # draw rotated rectangle on img
    # cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 2)

    # mask = cv2.GaussianBlur(mask, (31, 31), 0)
    #
    # cv2.imshow('sobel', cv2.resize(grad, (int(width / 4), int(height / 4))))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # mask = cv2.Canny(mask, 100, 200)
    # mask = cv2.threshold(mask, 64, 255, cv2.THRESH_BINARY)[1]
    # mask = cv2.Canny(mask, 100, 200)
    # mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    # mask = cv2.bilateralFilter(mask, 15, 80, 80)
    # Find contours
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # for cntr in contours:
    #     x, y, w, h = cv2.boundingRect(cntr)
    #     cv2.rectangle(cv_image, (x, y), (x + w, y + h), (36, 255, 12), 2)

    # x, y, w, h = cv2.boundingRect(max(contours, key=lambda c: cv2.arcLength(c, True)))

    cv2.imshow('mask', rect)
    cv2.imshow('result', cv2.resize(cv_image, (int(width / 4), int(height / 4))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return x, y, x + w, y + h  # border_left, border_top, border_right, border_bottom


def split(image):
    """
    Split raw image in half.

    :param image: raw image with two image areas and mostly black background.
    :return: left and right part.
    """
    width, height = image.size
    cut_point = int(width / 2)
    return raw.crop((0, 0, cut_point, height)), raw.crop((cut_point, 0, width, height))


# Size for result GIF image. ISO (Super) Duplex 120 has 23.5x25mm frame.
# 1000 height is typical maximum to autoplay GIF in social media.
output_size = (940, 1000)

# Get all jpg files in directory.
for file in Path(args.path).glob('*.jpg'):
    print('Process: %s ...' % file)
    with Image.open(file).convert(mode='RGB') as raw:
        # Part image in two halves.
        left, right = split(raw)
        # Search borders on left image.
        l_bl, l_bt, l_br, l_bb = search_borders(left)
        # Crop and resize for output.
        left = left.crop(
            [l_bl, l_bt, l_br, l_bb]) \
            .resize(output_size, resample=Image.Resampling.LANCZOS)
        # Search borders on right image. Only top and left will be used.
        r_bl, r_bt, _, _ = search_borders(right)
        r_width, r_height = right.size
        # Crop right image to be exactly the same size as left image, and resize for output.
        right = right.crop(
            [r_bl, l_bt, r_bl + (l_br - l_bl), l_bt + (l_bb - l_bt)]) \
            .resize(output_size, resample=Image.Resampling.LANCZOS)
    # Save as GIF
    left.save(
        format='GIF',
        fp=Path(file).with_suffix('.gif'),  # Use original file name, with new extension.
        save_all=True,  # Save appended images.
        append_images=[right],  # Add right image as second frame.
        optimize=True,  # Optimize pallet if it can be done.
        disposal=1,  # Use first frame as background. This prevent ripping on loop.
        duration=int(1 / 10 * 1000),  # 10 fps.
        loop=0)  # Infinite loop.
    # Tel user that we done with this image.
    print('Process: %s - OK' % file)
    # break

print('Ready!')

import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageFilter

parser = argparse.ArgumentParser(description="Convert ISO Duplex 120 stereos to gifs.")
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


def search_borders(image):
    """
    Naively search black borders on image.

    :param image: left or right half of original image.
    :return: cropping coordinates to remove black borders.
    """
    w, h = image.size
    # Search process params.
    threshold = 0.20  # Minimal percent of white pixels in row or column to determinate as image area.
    blur_size = 50  # Blur size to remove any small details.
    # Create single-bit mask for a border search.
    mask = ImageOps.autocontrast(  # Make equalized and blurred image high contrast.
        ImageOps.equalize(  # Equalize blurred image to make it normally bright.
            image.filter(  # Blur image to remove any small details.
                ImageFilter.GaussianBlur(radius=blur_size / 2)
            )
        ),
        cutoff=(20, 50)) \
        .convert(mode='1')  # Convert to a single-bit black and white.
    # Read pixels.
    px = mask.load()
    # Search all four borders. Clip for blurring.
    border_left = search_vertical_border_size(px, range(int(w / 2)), h, threshold) + blur_size
    border_top = search_horizontal_border_size(px, range(int(h / 2)), w, threshold) + blur_size
    border_right = search_vertical_border_size(px, range(w - 1, int(w / 2), -1), h, threshold) - blur_size
    border_bottom = search_horizontal_border_size(px, range(h - 1, int(h / 2), -1), w, threshold) - blur_size
    return border_left, border_top, border_right, border_bottom


def split(image):
    """
    Split raw image in half.

    :param image: raw image with two image areas and mostly black background.
    :return: left and right part.
    """
    width, height = image.size
    cut_point = int(width / 2)
    return raw.crop((0, 0, cut_point, height)), raw.crop((cut_point, 0, width, height))


# Size for result GIF image. ISO Duplex (Super) 120 has 23.5x25mm frame.
# 1000 height is typical maximum to autoplay GIF in social media.
output_size = (940, 1000)

# Get all jpg files in directory.
for file in Path(args.path).glob('*.jpg'):
    print('Process: %s ...' % file)
    with Image.open(file) as raw:
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

print('Ready!')

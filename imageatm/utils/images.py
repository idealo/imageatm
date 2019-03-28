import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from PIL import Image

IMG_FORMATS = ['JPEG', 'PNG']


def load_image(path: Path, target_size=None) -> Image:
    img = Image.open(path)
    f = img.format  # store format after opening as it gets lost after conversion

    if img.mode != 'RGB':
        # convert to RGBA first to avoid warning
        # we ignore alpha channel if available
        img = img.convert('RGBA').convert('RGB')

    if target_size:
        img = img.resize(target_size)

    img.format = f  # reassign format for later validation checks

    return img


def save_image(img, path: Path):
    img.convert('RGB').save(path)


def validate_image(
    file_name: Path, img_formats: List[str] = IMG_FORMATS
) -> Tuple[bool, Optional[Exception]]:
    """
    Checks whether File is valid image file:
        - file exists
        - file is readable
        - file is an image

    Args:
        file_name: Absolute path of file.

     Returns:
        True if file is valid image file.
        False else.
    """

    valid_image = False
    error = None

    try:
        img = load_image(file_name)

        if img.format in img_formats:
            img.load()  # Pillow uses lazy loading, so need to explicitly load

            valid_image = True

    except Exception as e:
        error = e

    return valid_image, error


def resize_image(img: Image, max_size: int, upscale: bool = False) -> Image:
    """Resizes image while keeping aspect ratio.

    The smaller dimension will be resized to max_size, i.e. a 400x500px image with max_size=300
    will be resized to 300x375px.

    Args:
        img: Pillow image object.
        max_size: Maximum width or height of resized image.
        upscale: If True will upscale small images to max_size.

     Returns:
        Pillow image object.
    """
    width, height = img.size

    if (max_size >= min(width, height)) and not upscale:
        return img

    min_dim = min(width, height)
    new_width = int((max_size / min_dim) * width)
    new_height = int((max_size / min_dim) * height)

    return img.resize((new_width, new_height))


def resize_image_mp(data_tuple: Tuple[str, str, str]):
    image_dir, new_image_dir, image_id = data_tuple
    img = load_image(Path(image_dir) / image_id)
    img = resize_image(img, max_size=300, upscale=False)
    save_image(img, Path(new_image_dir) / image_id)


def random_crop(img: np.array, crop_dims: Tuple[int, int]):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y : (y + ch), x : (x + cw), :]

from pathlib import Path
from imageatm.handlers import images as im

p = Path(__file__)

NON_EXISTENT_FILE_PATH = p.resolve().parent / '../data/test_images' / 'foo_bar.jpg'

INVALID_IMG_PATH = p.resolve().parent / '../data/test_images' / 'image_invalid.jpg'

TRUNCATED_IMG_PATH = p.resolve().parent / '../data/test_images' / 'truncated.jpg'

JPG_IMG_PATH = p.resolve().parent / '../data/test_images' / 'image_960x540.jpg'

PNG_IMG_PATH = p.resolve().parent / '../data/test_images' / 'image_png.png'

BMP_IMG_PATH = p.resolve().parent / '../data/test_images' / 'image_bmp.bmp'


def test_validate_image():
    valid, error = im.validate_image(NON_EXISTENT_FILE_PATH)
    assert valid is False
    assert isinstance(error, FileNotFoundError)

    valid, error = im.validate_image(INVALID_IMG_PATH)
    assert valid is False
    assert isinstance(error, OSError)

    valid, error = im.validate_image(TRUNCATED_IMG_PATH)
    assert valid is False
    assert isinstance(error, OSError)

    valid, error = im.validate_image(JPG_IMG_PATH)
    assert valid is True
    assert error is None

    valid, error = im.validate_image(PNG_IMG_PATH)
    assert valid is True
    assert error is None

    valid, error = im.validate_image(BMP_IMG_PATH)
    assert valid is False
    assert error is None


def test_resize_image():
    img_960x540 = im.load_image(JPG_IMG_PATH)

    img_pp = im.resize_image(img_960x540, max_size=300, upscale=False)
    w, h = img_pp.size
    assert w == 533
    assert h == 300

    img_pp = im.resize_image(img_960x540, max_size=300, upscale=True)
    w, h = img_pp.size
    assert w == 533
    assert h == 300

    img_pp = im.resize_image(img_960x540, max_size=600, upscale=False)
    w, h = img_pp.size
    assert w == 960
    assert h == 540

    img_pp = im.resize_image(img_960x540, max_size=600, upscale=True)
    w, h = img_pp.size
    assert w == 1066
    assert h == 600

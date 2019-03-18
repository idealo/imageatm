import logging
from pathlib import Path
from imageatm.utils.io import save_json, load_json


TEST_TARGET_FILE = Path('./tests/data/test_samples/test_target_file.json').resolve()


class TestIo(object):
    def test_save_json(self):
        data = [
            {'image_id': 'helmet_1.jpg', 'label': 'left'},
            {'image_id': 'helmet_2.jpg', 'label': 'left'},
            {'image_id': 'image_png.png', 'label': 'right'},
        ]

        target_file = TEST_TARGET_FILE
        assert target_file.exists() is False
        save_json(data, target_file)
        assert target_file.exists()
        assert load_json(target_file) == data
        target_file.unlink()

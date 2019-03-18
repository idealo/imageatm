import mock
from imageatm.handlers.keras_utils import use_multiprocessing


class TestKerasUtils(object):

    @mock.patch('imageatm.handlers.keras_utils._get_available_gpus', return_value=[])
    def test__use_multiprocessing_false(self, mock):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == False
        assert num_worker == 1

    @mock.patch('imageatm.handlers.keras_utils.cpu_count', return_value=4711)
    @mock.patch('imageatm.handlers.keras_utils._get_available_gpus', return_value=['foo', 'bar'])
    def test__use_multiprocessing_true(self, mock1, mock2):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == True
        assert num_worker == 4711

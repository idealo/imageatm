import mock
from imageatm.utils.tf_keras import use_multiprocessing


class TestTfKeras(object):
    @mock.patch('imageatm.utils.tf_keras._get_available_gpus', return_value=[])
    def test__use_multiprocessing_false(self, mock):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == False
        assert num_worker == 1

    @mock.patch('imageatm.utils.tf_keras.cpu_count', return_value=4711)
    @mock.patch('imageatm.utils.tf_keras._get_available_gpus', return_value=['foo', 'bar'])
    def test__use_multiprocessing_true(self, mock1, mock2):
        use_multi, num_worker = use_multiprocessing()

        assert use_multi == True
        assert num_worker == 4711

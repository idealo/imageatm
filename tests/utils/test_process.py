import logging
import pytest
from imageatm.utils.process import run_cmd


class TestProcess(object):
    def test_run_cmd_1(self, mocker):
        mp_debug = mocker.patch('logging.Logger.debug')
        mp_info = mocker.patch('logging.Logger.info')

        cmd = 'echo Hello world'
        logger = logging.Logger(__name__)
        level = 'debug'
        return_output = False

        assert run_cmd(cmd, logger, level, return_output) == None

        mp_debug.assert_called_once()
        mp_info.assert_not_called()

    def test_run_cmd_2(self, mocker):
        mp_debug = mocker.patch('logging.Logger.debug')
        mp_info = mocker.patch('logging.Logger.info')

        cmd = 'echo Hello world'
        logger = logging.Logger(__name__)
        level = 'info'
        return_output = False

        run_cmd(cmd, logger, level, return_output)

        mp_debug.assert_not_called()
        mp_info.assert_called_once()

    def test_run_cmd_3(self, mocker):
        mp_debug = mocker.patch('logging.Logger.debug')
        mp_info = mocker.patch('logging.Logger.info')

        cmd = 'echo Hello world'
        logger = logging.Logger(__name__)
        level = 'debug'
        return_output = True

        assert run_cmd(cmd, logger, level, return_output) == 'Hello world'
        mp_debug.assert_called_once()
        mp_info.assert_not_called()

    def test_run_cmd_4(self, mocker):
        mp_debug = mocker.patch('logging.Logger.debug')
        mp_info = mocker.patch('logging.Logger.info')
        mp_error = mocker.patch('logging.Logger.error')

        cmd = 'echo2 Hello world'
        logger = logging.Logger(__name__)
        level = 'debug'
        return_output = False

        with pytest.raises(Exception) as excinfo:
            run_cmd(cmd, logger, level, return_output)
            mp_debug.assert_not_called()
            mp_info.assert_not_called()
            mp_error.assert_called_once()

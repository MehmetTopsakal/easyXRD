import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import easyxrd
from easyxrd._gsas import load_gsasii
from easyxrd.diagnostics import collect_diagnostics


class StartupTests(unittest.TestCase):
    def test_import_is_quiet_and_does_not_load_scientific_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, HOME=directory, USERPROFILE=directory)
            result = subprocess.run([sys.executable, '-c',
                'import easyxrd, sys; '
                'assert "numpy" not in sys.modules; '
                'assert "GSASII" not in sys.modules'],
                env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout, '')
            self.assertEqual(result.stderr, '')
            self.assertFalse((Path(directory) / '.easyxrd_scratch').exists())

    def test_key_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'key'
            self.assertEqual(easyxrd._read_api_key(path), 'not found')
            for contents, expected in [('', 'invalid'), (' \n', 'invalid'),
                                        ('short', 'invalid'), ('a' * 32, 'a' * 32),
                                        ('label ' + 'b' * 32, 'b' * 32)]:
                path.write_text(contents)
                self.assertEqual(easyxrd._read_api_key(path), expected)
            path.write_bytes(b'\xff')
            self.assertEqual(easyxrd._read_api_key(path), 'invalid')
            with patch('pathlib.Path.read_text', side_effect=PermissionError):
                self.assertEqual(easyxrd._read_api_key(path), 'invalid')

    def test_key_is_fully_redacted(self):
        with patch.dict(easyxrd.easyxrd_defaults, mp_api_key='secret-value'):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                easyxrd.print_defaults()
            self.assertNotIn('secret', out.getvalue())
            self.assertIn('<configured>', out.getvalue())

    def test_hidden_prints_restores_stdout_on_exception(self):
        original = sys.stdout
        with self.assertRaises(ValueError):
            with easyxrd.HiddenPrints():
                raise ValueError('test')
        self.assertIs(sys.stdout, original)

    def test_diagnostics_without_numpy(self):
        with patch('easyxrd.diagnostics.import_module', side_effect=ImportError('ABI')):
            result = collect_diagnostics()
        self.assertIn('ABI', result['numpy_import_error'])
        self.assertEqual(result['python_executable'], sys.executable)
        json.dumps(result)


class GsasTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.dict(easyxrd.easyxrd_defaults, gsasii_lib_path='not found')
        self.config.start()
        self.addCleanup(self.config.stop)

    def test_modern_backend_can_be_loaded_repeatedly(self):
        backend = types.SimpleNamespace(__file__='/installed/GSASII/GSASIIscriptable.py')
        with patch('easyxrd._gsas.importlib.import_module', return_value=backend) as importer:
            self.assertIs(load_gsasii(), backend)
            self.assertIs(load_gsasii(), backend)
            self.assertEqual(importer.call_args.args, ('GSASII.GSASIIscriptable',))

    def test_invalid_explicit_path_fails_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_gsasii(Path(directory) / 'absent')

    def test_failed_explicit_import_restores_search_path(self):
        previous = sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            with patch('easyxrd._gsas.importlib.import_module', side_effect=RuntimeError('ABI')):
                with self.assertRaisesRegex(ImportError, 'ABI'):
                    load_gsasii(directory)
        self.assertEqual(sys.path, previous)

    def test_explicit_path_has_priority(self):
        previous = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), previous))
        with tempfile.TemporaryDirectory() as directory:
            backend = types.SimpleNamespace(__file__=str(Path(directory) / 'GSASIIscriptable.py'))
            with patch('easyxrd._gsas.importlib.import_module', return_value=backend) as importer:
                self.assertIs(load_gsasii(directory), backend)
                self.assertEqual(sys.path[0], str(Path(directory).resolve()))
                importer.assert_called_once_with('GSASIIscriptable')

    def test_missing_backend_has_actionable_error(self):
        missing = ModuleNotFoundError("No module named 'GSASII'", name='GSASII')
        with patch('easyxrd._gsas.importlib.import_module', side_effect=[missing, ImportError('missing')]):
            with self.assertRaisesRegex(ImportError, 'gsasii_lib_path'):
                load_gsasii()

    def test_internal_missing_dependency_does_not_try_other_backend(self):
        missing = ModuleNotFoundError("No module named 'pyspg'", name='pyspg')
        with patch('easyxrd._gsas.importlib.import_module', side_effect=missing) as importer:
            with self.assertRaisesRegex(ImportError, 'pyspg'):
                load_gsasii()
            self.assertEqual(importer.call_count, 1)


if __name__ == '__main__':
    unittest.main()

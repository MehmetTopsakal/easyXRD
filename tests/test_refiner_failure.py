"""Test the actual setup failure path without scientific dependencies."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


class RefinerFailureTests(unittest.TestCase):
    def test_backend_failure_preserves_existing_state(self):
        source = Path(__file__).resolve().parents[1] / 'easyxrd' / 'core.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'exrd')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'setup_gsas2_refiner')
        loader = Mock(side_effect=ImportError('backend unavailable'))
        namespace = {'load_gsasii': loader}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        state = type('State', (), {})()
        state.ds = {'i1d_refined': object(), 'i1d_gsas_background': object()}
        state.gpx = object()
        state.gsasii_lib_path = '/existing/backend'
        before = dict(state.__dict__)
        dataset_before = dict(state.ds)
        with self.assertRaisesRegex(ImportError, 'backend unavailable'):
            namespace['setup_gsas2_refiner'](state, gsasii_lib_path='/requested/backend')
        loader.assert_called_once_with('/requested/backend')
        self.assertEqual(state.__dict__, before)
        self.assertEqual(state.ds, dataset_before)

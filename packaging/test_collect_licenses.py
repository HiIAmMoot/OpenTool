import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'collect_licenses', Path(__file__).with_name('collect_licenses.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class LicenseTests(unittest.TestCase):
    def test_missing_distribution_licenses_are_all_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'licenses'
            with patch.object(collector.metadata, 'files', return_value=[]):
                with self.assertRaises(RuntimeError) as raised:
                    collector.collect(destination)
            for name in collector.DISTRIBUTIONS:
                self.assertIn(name, str(raised.exception))
            self.assertFalse(destination.exists())

    @unittest.skipUnless(Path(sys.prefix).name == 'venv' and Path(sys.prefix).parent.name == 'build',
                         'needs the build venv (build/venv) with the packaged distributions installed')
    def test_real_build_environment_licenses_are_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'licenses'
            collector.collect(destination)
            for name in collector.DISTRIBUTIONS:
                self.assertTrue(any((destination / name).rglob('*')), name)
            # These transitive imports are present in the frozen PYZ, not just the build venv.
            for name in ('packaging',):
                self.assertTrue(any((destination / name).rglob('*')), name)
            for name in ('python/LICENSE.txt', 'tcltk/tcl-license.terms',
                         'tcltk/tk-license.terms', 'oidn/LICENSE.txt'):
                self.assertTrue((destination / name).is_file(), name)
            self.assertEqual(len(list((destination / 'oidn').glob('third-party-programs*.txt'))), 3)


if __name__ == '__main__':
    unittest.main()

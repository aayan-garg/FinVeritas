"""
FinVeritas Smoke Tests:
Verifies basic environment sanity and package layout importability.
"""

import sys
import unittest
from pathlib import Path

# Ensure src/ is in sys.path for direct unittest execution without prior editable install
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class TestFinVeritasSmoke(unittest.TestCase):
    def test_import_finveritas_root(self):
        import finveritas

        self.assertTrue(hasattr(finveritas, "__version__"))
        self.assertEqual(finveritas.__version__, "0.1.0")

    def test_import_subpackages(self):
        import finveritas.api
        import finveritas.crag_engine
        import finveritas.nlp_evasion
        import finveritas.rag_indexing
        import finveritas.verification

        self.assertIsNotNone(finveritas.nlp_evasion)
        self.assertIsNotNone(finveritas.rag_indexing)
        self.assertIsNotNone(finveritas.crag_engine)
        self.assertIsNotNone(finveritas.verification)
        self.assertIsNotNone(finveritas.api)


if __name__ == "__main__":
    unittest.main()

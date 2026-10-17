import tempfile
import unittest
from pathlib import Path
import numpy as np
from nclt_binary import HOKUYO_DTYPE,read_checked_hokuyo


class FixedLaserTests(unittest.TestCase):
    def records(self,times):
        rows=np.zeros(len(times),dtype=HOKUYO_DTYPE)
        rows['timestamp']=times
        rows['ranges']=np.arange(1081,dtype=np.uint16)
        return rows

    def read(self,raw):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scan.bin';path.write_bytes(raw)
            return read_checked_hokuyo(path,[1300000000000000,1300000000100000],
                                      [1300000000000000,1300000000100000])

    def test_canonical_records_preserve_ranges_and_timestamps(self):
        rows=self.records([1300000000000000,1300000000025000])
        data,audit=self.read(rows.tobytes())
        np.testing.assert_array_equal(data,rows)
        self.assertEqual(audit['invalid_suffix_records'],0)

    def test_malformed_complete_suffix_and_partial_bytes_are_audited(self):
        rows=self.records([1300000000000000,1300000000025000,5684755367836471014])
        data,audit=self.read(rows.tobytes()+b'x'*600)
        np.testing.assert_array_equal(data,rows[:2])
        self.assertEqual(audit['invalid_suffix_records'],1)
        self.assertEqual(audit['incomplete_trailing_bytes'],600)

    def test_interior_corruption_is_not_silently_removed(self):
        rows=self.records([1300000000000000,5684755367836471014,1300000000050000])
        with self.assertRaisesRegex(ValueError,'interior'):self.read(rows.tobytes())

    def test_nonmonotonic_records_are_rejected(self):
        rows=self.records([1300000000050000,1300000000025000])
        with self.assertRaisesRegex(ValueError,'strictly increasing'):self.read(rows.tobytes())


if __name__=='__main__':unittest.main()

import copy
import hashlib
import unittest

from falcon.reviewed_redactions import audit_ledger, validate_ledger, LedgerValidationError


class ReviewedRedactionsTests(unittest.TestCase):
    def setUp(self):
        digest = hashlib.sha256(b"exact-source-bytes").hexdigest()
        self.expected = {"records": [
            {"image_id": "a", "source_sha256": digest, "bbox": [0, 0, 10, 10]}
        ]}
        self.baseline = {"records": [
            {"image_id": "a", "redactions": [[1, 4, 9, 9]]}
        ]}
        self.ledger = {"reviewed": True, "records": [
            {"image_id": "a", "source_sha256": digest, "bbox": [0, 0, 10, 10],
             "rectangles": [[1, 4, 5, 9], [5, 4, 9, 9], [0, 0, 2, 2]],
             "status": "reviewed", "reviewer": "reviewer-1"}
        ]}

    def test_good_split_union_is_exact_and_monotone(self):
        report = validate_ledger(self.expected, self.baseline, self.ledger)
        self.assertTrue(report["valid"])
        self.assertEqual(report["reviewed"], 1)

    def test_corrupt_hash_fails_closed(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["records"][0]["source_sha256"] = "0" * 64
        with self.assertRaises(LedgerValidationError):
            validate_ledger(self.expected, self.baseline, ledger)

    def test_missing_and_extra_ids_are_reported(self):
        missing = copy.deepcopy(self.ledger)
        missing["records"] = []
        self.assertIn("ledger missing IDs", " ".join(audit_ledger(self.expected, self.baseline, missing)["errors"]))
        extra = copy.deepcopy(self.ledger)
        extra["records"].append({"image_id": "b"})
        self.assertIn("ledger extra IDs", " ".join(audit_ledger(self.expected, self.baseline, extra)["errors"]))

    def test_reopened_pixel_strip_fails_monotonicity(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["records"][0]["rectangles"] = [[1, 4, 5, 9], [6, 4, 9, 9]]
        with self.assertRaisesRegex(LedgerValidationError, "not fully covered"):
            validate_ledger(self.expected, self.baseline, ledger)

    def test_invalid_rectangle_and_unreviewed_status_fail(self):
        ledger = copy.deepcopy(self.ledger)
        ledger["records"][0]["rectangles"] = [[1, 4, 1, 9]]
        ledger["records"][0]["status"] = "pending"
        report = audit_ledger(self.expected, self.baseline, ledger)
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()

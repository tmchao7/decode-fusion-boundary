import sys
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from nanovllm.backends.megakernel import audit, schedule as tg  # noqa: E402
from tools import verify_schedule  # noqa: E402


class TaskGraphAuditTests(unittest.TestCase):
    def test_supported_matrix_has_no_cross_sm_write_collisions(self):
        for batch_size in (1, 2, 4, 8):
            for num_sms in (1, 4, 108, 170):
                for attn_splits in (1, 2):
                    with self.subTest(
                        batch_size=batch_size,
                        num_sms=num_sms,
                        attn_splits=attn_splits,
                    ):
                        geo = tg.default_geometry(batch_size, attn_splits=attn_splits)
                        sched = tg.build_schedule(geo, num_sms)
                        report = audit.audit_schedule(
                            sched,
                            geometry=geo,
                            batch_size=batch_size,
                            num_sms=num_sms,
                            attn_splits=attn_splits,
                        )
                        self.assertTrue(report.ok, report.describe())
                        self.assertGreater(report.checked_writes, 0)

    def test_verify_matrix_counts_audit_failure_as_configuration_failure(self):
        failed = audit.AuditReport(False, ("synthetic audit failure",), 0)
        with mock.patch.object(verify_schedule.A, "audit_schedule", return_value=failed):
            payload = verify_schedule.verify_matrix(
                batch_sizes=[1],
                num_sms_values=[1],
                fusion_groups=[1],
                attn_splits=[1],
                sample=1,
            )
        self.assertEqual(payload["totals"]["passed"], 0)
        self.assertEqual(payload["totals"]["failed"], 1)
        self.assertFalse(payload["configs"][0]["ok"])
        self.assertIn("synthetic audit failure", payload["configs"][0]["failures"])

    def test_packed_generic_gemm_range_must_fit_matrix(self):
        geo = tg.default_geometry(1)
        sched = tg.build_schedule(geo, 1)
        flat = list(sched.flat_inst_buf)
        # First O_PROJ instruction is a packed range. Corrupt its start so the
        # declared range runs past the 1024-wide output matrix.
        width = tg.TG_INST_WIDTH
        for offset in range(0, len(flat), width):
            if flat[offset] == tg.TgOp.O_PROJ:
                flat[offset + 10] = (geo.hidden // 16) - 1
                break
        broken = replace(sched, flat_inst_buf=flat)
        report = audit.audit_schedule(
            broken, geometry=geo, batch_size=1, num_sms=1, attn_splits=1
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("O_PROJ" in failure and "range" in failure for failure in report.failures))

    def test_attn_split_index_must_fit_declared_geometry(self):
        geo = tg.default_geometry(1, attn_splits=2)
        sched = tg.build_schedule(geo, 1)
        flat = list(sched.flat_inst_buf)
        width = tg.TG_INST_WIDTH
        for offset in range(0, len(flat), width):
            if flat[offset] == tg.TgOp.ATTN_DECODE:
                flat[offset + 4] = 2
                break
        broken = replace(sched, flat_inst_buf=flat)
        report = audit.audit_schedule(
            broken, geometry=geo, batch_size=1, num_sms=1, attn_splits=2
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("split" in failure for failure in report.failures))

    def test_qkv_pack_must_preserve_one_kv_bar_owner(self):
        geo = tg.default_geometry(1)
        sched = tg.build_schedule(geo, 1)
        flat = list(sched.flat_inst_buf)
        width = tg.TG_INST_WIDTH
        for offset in range(0, len(flat), width):
            if flat[offset] == tg.TgOp.QKV_PROJ:
                # Keep the range in bounds but make it cross the QKV head
                # boundary; one arrival cannot represent two bar owners.
                flat[offset + 10] = 15
                flat[offset + 9] = 2
                break
        broken = replace(sched, flat_inst_buf=flat)
        report = audit.audit_schedule(
            broken, geometry=geo, batch_size=1, num_sms=1, attn_splits=1
        )
        self.assertFalse(report.ok)
        self.assertTrue(any("QKV" in failure and "bar" in failure for failure in report.failures))


if __name__ == "__main__":
    unittest.main()

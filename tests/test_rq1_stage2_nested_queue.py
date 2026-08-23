import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rq1_stage2_nested_queue.py"
SPEC = importlib.util.spec_from_file_location("run_rq1_stage2_nested_queue", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class RQ1Stage2NestedQueueTests(unittest.TestCase):
    def make_layout(self, root: Path) -> tuple[Path, Path, Path, Path, dict[int, int]]:
        output = root / "output"
        manifests = root / "manifests"
        derangements = root / "derangements"
        data_root = root / "data"
        for outer in range(7):
            fold = manifests / f"outer_{outer}"
            fold.mkdir(parents=True)
            (fold / "building_threshold.json").write_text('{"selected_threshold": 0.7}\n')
        return output, manifests, derangements, data_root, {outer: 6000 for outer in range(7)}

    def write_training_artifacts(self, run_dir: Path, step: int = 6000) -> None:
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "completed.json").write_text(
            f'{{"status": "completed", "optimizer_steps_completed": {step}}}\n'
        )
        (run_dir / "run_info.json").write_text(
            "{"
            '"fixed_step_mode": true, '
            f'"max_optimizer_steps": {step}, '
            '"fixed_step_iterator_policy": "persistent_full_pass_cycle", '
            '"fixed_step_training_diagnostics_policy": "full_at_validation_loss_at_heartbeat", '
            '"validation_batch_limit": null'
            "}\n"
        )
        (run_dir / "metrics_history.csv").write_text(f"epoch,metric\n{step},0.5\n")
        (run_dir / "checkpoints" / f"step_{step:06d}.pth").write_bytes(b"checkpoint")

    def write_evaluation_artifacts(self, eval_dir: Path, step: int = 6000) -> None:
        eval_dir.mkdir(parents=True)
        (eval_dir / "event_generalization.json").write_text('{"event_count": 2}\n')
        (eval_dir / "metrics.json").write_text(
            f'{{"sample_count": 2, "checkpoint_epoch": {step}, "split": "val"}}\n'
        )
        (eval_dir / "per_event_metrics.csv").write_text("event_id\na\nb\n")
        (eval_dir / "sample_metrics.csv").write_text("id\n1\n2\n")

    def test_builds_all_70_final_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, manifests, derangements, data_root, selection = self.make_layout(Path(tmp))
            tasks, completed = QUEUE.build_final_tasks(
                output, manifests, derangements, data_root, Path(tmp), selection
            )
            self.assertEqual(completed, 0)
            self.assertEqual(len(tasks), 70)
            self.assertEqual(len({task["id"] for task in tasks}), 70)
            self.assertEqual(len(list((output / "configs").glob("final_*.yaml"))), 70)
            self.assertIn("final_o0_C0_s42", {task["id"] for task in tasks})
            self.assertIn("final_o6_C3_s2026", {task["id"] for task in tasks})

    def test_skips_only_fully_completed_final_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, manifests, derangements, data_root, selection = self.make_layout(Path(tmp))
            run_dir = output / "final" / "outer_0" / "C0" / "seed_42"
            eval_dir = output / "evaluation" / "outer_0" / "C0" / "seed_42"
            self.write_training_artifacts(run_dir)
            self.write_evaluation_artifacts(eval_dir)
            tasks, completed = QUEUE.build_final_tasks(
                output, manifests, derangements, data_root, Path(tmp), selection
            )
            self.assertEqual(completed, 1)
            self.assertEqual(len(tasks), 69)
            self.assertNotIn("final_o0_C0_s42", {task["id"] for task in tasks})

    def test_rejects_partial_final_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, manifests, derangements, data_root, selection = self.make_layout(Path(tmp))
            run_dir = output / "final" / "outer_0" / "C0" / "seed_42"
            run_dir.mkdir(parents=True)
            (run_dir / "latest_metrics.json").write_text('{}\n')
            with self.assertRaisesRegex(RuntimeError, "partial or invalid run requires review"):
                QUEUE.build_final_tasks(
                    output, manifests, derangements, data_root, Path(tmp), selection
                )

    def test_recovers_completed_training_with_evaluation_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, manifests, derangements, data_root, selection = self.make_layout(Path(tmp))
            run_dir = output / "final" / "outer_0" / "C0" / "seed_42"
            self.write_training_artifacts(run_dir)
            tasks, completed = QUEUE.build_final_tasks(
                output, manifests, derangements, data_root, Path(tmp), selection
            )
            self.assertEqual(completed, 0)
            recovered = next(task for task in tasks if task["id"] == "final_o0_C0_s42_eval_only")
            command = recovered["command"][-1]
            self.assertIn("test_stage2_v2.py", command)
            self.assertNotIn("train_stage2_v2.py", command)
            self.assertIn(".in_progress.", command)

    def test_rejects_marker_only_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output, manifests, derangements, data_root, selection = self.make_layout(Path(tmp))
            run_dir = output / "final" / "outer_0" / "C0" / "seed_42"
            eval_dir = output / "evaluation" / "outer_0" / "C0" / "seed_42"
            self.write_training_artifacts(run_dir)
            eval_dir.mkdir(parents=True)
            (eval_dir / "event_generalization.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "partial or invalid evaluation"):
                QUEUE.build_final_tasks(
                    output, manifests, derangements, data_root, Path(tmp), selection
                )

    def test_numerical_completion_is_optional_but_fail_closed_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            eval_dir = root / "eval"
            self.write_training_artifacts(run_dir)
            self.write_evaluation_artifacts(eval_dir)
            self.assertTrue(QUEUE.training_complete(run_dir, 6000))
            self.assertFalse(
                QUEUE.training_complete(
                    run_dir, 6000, require_numerical_integrity=True
                )
            )
            completion = run_dir / "completed.json"
            completion.write_text(
                '{"status":"completed","optimizer_steps_completed":6000,'
                '"numerical_integrity_passed":true}\n'
            )
            (run_dir / "numerical_integrity.json").write_text('{"status":"passed"}\n')
            checkpoint = run_dir / "checkpoints" / "step_006000.pth"
            checkpoint.with_suffix(".pth.integrity.json").write_text('{"status":"passed"}\n')
            (eval_dir / "numerical_integrity.json").write_text('{"status":"passed"}\n')
            self.assertTrue(
                QUEUE.training_complete(
                    run_dir, 6000, require_numerical_integrity=True
                )
            )
            self.assertTrue(
                QUEUE.evaluation_complete(
                    eval_dir, 6000, require_numerical_integrity=True
                )
            )


if __name__ == "__main__":
    unittest.main()

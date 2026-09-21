import json

import run_vmod_bidirectional_after_margin as runner


def test_freeze_calibrated_candidate_preserves_exact_artifacts(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    files = {
        "teacher/summary.json": b"{}",
        "teacher/steps.csv": b"a,b\n1,2\n",
        "teacher/episodes.csv": b"a,b\n3,4\n",
        "dataset/opf_imitation_dataset.npz": b"dataset",
        "dataset/manifest.json": b"{}",
    }
    for seed in runner.SEEDS:
        files[f"policies/policy_init_seed{seed}.pth"] = f"policy-{seed}".encode()
        files[f"policies/summary_seed{seed}.json"] = json.dumps(
            {"seed": seed}
        ).encode()
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    manifest = runner.freeze_calibrated_candidate(source, destination)

    assert manifest["frozen_artifact_count"] == len(files)
    for relative, content in files.items():
        assert (destination / relative).read_bytes() == content
    stored = json.loads(
        (destination / "frozen_margin_candidate.json").read_text(encoding="utf-8")
    )
    assert stored["frozen_artifact_count"] == len(files)


def test_freeze_rejects_existing_artifact_with_different_hash(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    required = {
        "teacher/summary.json": b"{}",
        "teacher/steps.csv": b"steps",
        "teacher/episodes.csv": b"episodes",
        "dataset/opf_imitation_dataset.npz": b"dataset",
        "dataset/manifest.json": b"{}",
    }
    for seed in runner.SEEDS:
        required[f"policies/policy_init_seed{seed}.pth"] = b"policy"
        required[f"policies/summary_seed{seed}.json"] = b"{}"
    for relative, content in required.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    conflict = destination / "dataset" / "manifest.json"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b"different")

    try:
        runner.freeze_calibrated_candidate(source, destination)
    except RuntimeError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("Expected a hash mismatch")

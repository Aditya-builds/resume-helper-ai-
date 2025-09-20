import os
import json
import shutil
from utils.migration import migrate_resumes_to_scores


def setup_test_env(base_path):
    resumes = os.path.join(base_path, "resumes")
    scores = os.path.join(base_path, "scores")
    # ensure clean
    if os.path.exists(resumes):
        shutil.rmtree(resumes)
    if os.path.exists(scores):
        shutil.rmtree(scores)
    os.makedirs(os.path.join(resumes, "job_test"), exist_ok=True)
    return resumes, scores


def test_migration_moves_and_writes_marker(tmp_path):
    # Use a temporary workspace root
    base = str(tmp_path)
    resumes_path, scores_path = setup_test_env(base)

    # Write a small scoring_results.json into resumes/job_test
    sample = [{
        "resume_file": "job_test_resume_1.pdf",
        "score": 42,
        "verdict": "Good",
        "details": {}
    }]
    old_file = os.path.join(resumes_path, "job_test", "scoring_results.json")
    with open(old_file, "w", encoding="utf-8") as f:
        json.dump(sample, f)

    # Run migration
    summary = migrate_resumes_to_scores(resumes_path, scores_path)

    # Assertions
    target_dir = os.path.join(scores_path, "job_test")
    target_file = os.path.join(target_dir, "scoring_results.json")
    marker_file = os.path.join(target_dir, "last_score_location.txt")

    assert os.path.exists(target_file), "scoring_results.json was not moved to scores/"
    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data[0]["score"] == 42

    assert os.path.exists(marker_file), "last_score_location.txt was not written"
    with open(marker_file, "r", encoding="utf-8") as mf:
        loc = mf.read().strip()
    assert os.path.abspath(target_file) == loc

    # summary should indicate a move
    assert "job_test" in summary.get("moved", []) or "job_test" in summary.get("merged", [])

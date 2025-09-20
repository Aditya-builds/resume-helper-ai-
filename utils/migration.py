import os
import json

def migrate_resumes_to_scores(resumes_path: str, scores_path: str) -> dict:
    """Move or merge any scoring_results.json files from resumes/<job_id>/ to scores/<job_id>/.

    Returns a dict summary with keys:
      - moved: list of job_ids moved
      - merged: list of job_ids merged
      - errors: list of (job_id, error)
    """
    summary = {"moved": [], "merged": [], "errors": []}
    os.makedirs(resumes_path, exist_ok=True)
    os.makedirs(scores_path, exist_ok=True)

    for resume_subdir in os.listdir(resumes_path):
        resume_dir_path = os.path.join(resumes_path, resume_subdir)
        if not os.path.isdir(resume_dir_path):
            continue
        old_scores_file = os.path.join(resume_dir_path, "scoring_results.json")
        if os.path.exists(old_scores_file):
            target_scores_dir = os.path.join(scores_path, resume_subdir)
            os.makedirs(target_scores_dir, exist_ok=True)
            target_scores_file = os.path.join(target_scores_dir, "scoring_results.json")
            try:
                if not os.path.exists(target_scores_file):
                    os.replace(old_scores_file, target_scores_file)
                    summary["moved"].append(resume_subdir)
                    try:
                        with open(os.path.join(target_scores_dir, "last_score_location.txt"), "w", encoding="utf-8") as lf:
                            lf.write(os.path.abspath(target_scores_file))
                    except Exception:
                        pass
                else:
                    # merge
                    try:
                        with open(old_scores_file, "r", encoding="utf-8") as of:
                            old_data = json.load(of)
                    except Exception:
                        old_data = []
                    try:
                        with open(target_scores_file, "r", encoding="utf-8") as tf:
                            target_data = json.load(tf)
                    except Exception:
                        target_data = []
                    existing_files = {entry.get("resume_file"): entry for entry in target_data}
                    for entry in old_data:
                        if entry.get("resume_file") not in existing_files:
                            target_data.append(entry)
                    try:
                        with open(target_scores_file, "w", encoding="utf-8") as tf:
                            json.dump(target_data, tf, indent=4)
                    except Exception:
                        pass
                    try:
                        os.remove(old_scores_file)
                    except Exception:
                        pass
                    summary["merged"].append(resume_subdir)
            except Exception as e:
                summary["errors"].append((resume_subdir, str(e)))

    return summary

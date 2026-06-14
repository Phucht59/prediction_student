import subprocess
import os

repo_dir = r"c:\Huflit\kltn"
output_dir = r"c:\Huflit\kltn\.agents\auditor_milestone4_1"

def run_git(args):
    res = subprocess.run(["git"] + args, cwd=repo_dir, capture_output=True, text=True)
    return res.stdout, res.stderr

# 1. git status
status_out, status_err = run_git(["status"])
with open(os.path.join(output_dir, "git_status.txt"), "w", encoding="utf-8") as f:
    f.write("=== STATUS ===\n")
    f.write(status_out)
    f.write("\n=== ERR ===\n")
    f.write(status_err)

# 2. git diff for data_pipeline.py (working tree vs HEAD)
diff_data_out, _ = run_git(["diff", "src/data_pipeline.py"])
with open(os.path.join(output_dir, "diff_data_pipeline_wt.txt"), "w", encoding="utf-8") as f:
    f.write(diff_data_out)

# 3. git diff for train_pipeline.py (working tree vs HEAD)
diff_train_out, _ = run_git(["diff", "src/train_pipeline.py"])
with open(os.path.join(output_dir, "diff_train_pipeline_wt.txt"), "w", encoding="utf-8") as f:
    f.write(diff_train_out)

# 4. git diff from origin/temp-main to HEAD for both files
diff_data_origin, _ = run_git(["diff", "origin/temp-main", "HEAD", "--", "src/data_pipeline.py"])
with open(os.path.join(output_dir, "diff_data_pipeline_origin.txt"), "w", encoding="utf-8") as f:
    f.write(diff_data_origin)

diff_train_origin, _ = run_git(["diff", "origin/temp-main", "HEAD", "--", "src/train_pipeline.py"])
with open(os.path.join(output_dir, "diff_train_pipeline_origin.txt"), "w", encoding="utf-8") as f:
    f.write(diff_train_origin)

# 5. git log with patches for both files
log_data, _ = run_git(["log", "-p", "--", "src/data_pipeline.py"])
with open(os.path.join(output_dir, "log_data_pipeline.txt"), "w", encoding="utf-8") as f:
    f.write(log_data)

log_train, _ = run_git(["log", "-p", "--", "src/train_pipeline.py"])
with open(os.path.join(output_dir, "log_train_pipeline.txt"), "w", encoding="utf-8") as f:
    f.write(log_train)

print("Git analysis complete!")

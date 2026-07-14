import subprocess, sys

def run(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out)
    if check and r.returncode != 0:
        sys.exit("FAILED")
    return r

run("add", "-A")
run("commit", "-q", "-F", ".git/CM_rename.txt")
run("log", "--oneline", "-1")

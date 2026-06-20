import subprocess

for script in ['truth_obs.py', 'partfilt.py', 'enkf.py', 'plots.py']:
    print(f"\n=== running {script} ===")
    subprocess.run(['python', script], check=True)

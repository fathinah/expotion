import time
import subprocess

# Wait for 8 hours (8 hours * 60 minutes * 60 seconds)
time.sleep(8 * 60 * 60)

# Execute the train.py script
subprocess.run(["python", "train.py"])


import subprocess
file_name = '.\hw1.py'
iterate = 50
for i in range(iterate):
    print(f"Running hw1.py {i+1} time ..")
    subprocess.run(['python3',file_name])

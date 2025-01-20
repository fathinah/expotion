import socket
import subprocess
import signal

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0
def run_tensorboard(logdir, port=6006, host="0.0.0.0"):
    # try:
    #     output = subprocess.check_output(f"lsof -i:{port}", shell=True).decode()
    #     if output:
    #         print(f"Port {port} is already in use. TensorBoard might already be running.")
    #         return
    # except subprocess.CalledProcessError:
    #     pass
    t_command = f'tensorboard --logdir {logdir} --host 0.0.0.0 --port {port}'


if __name__ == "__main__":
    logdir = "exp/face_only"  # Replace with your log directory path
    port = 6006                # Replace with your desired port
    run_tensorboard(logdir, port)
    print("port ready for tensorboard")
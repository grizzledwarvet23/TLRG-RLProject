import numpy as np
import torch
import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5065
MESSAGE = "Hello, UNITY!"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))
import socket
import numpy
import time
import cv2

UDP_IP = "127.0.0.1"
UDP_PORT = 999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))


s=b''

while True:

  data, addr = sock.recvfrom(6) # b'eframe' is 6 byte long

  if data == b'sframe':
    while True:
      sock.sendto(b'n', addr)
      data, addr = sock.recvfrom(46080)
      if data == b'eframe':
        break
      s += data

  if len(s) == (46080*20):

    frame = numpy.fromstring (s,dtype=numpy.uint8)
    frame = frame.reshape (480,640,3)

    cv2.imshow('frame',frame)

    s=b''

  if cv2.waitKey(1) & 0xFF == ord ('q'):
    break

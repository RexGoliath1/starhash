import socket
import numpy as np
import cv2

UDP_IP = '127.0.0.1'
UDP_PORT = 999
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
def xrange(x):
  return iter(range(x))

while (True):
  ret, frame = cap.read()
  if not ret:
    print("No frame")
  cv2.imshow('frame', frame)
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  d = frame.flatten()
  s = d.tostring()

  sock.sendto(b'sframe', (UDP_IP, UDP_PORT))
  for i in xrange(20):
    data, addr = sock.recvfrom(1) # b'n' is 1 byte long
    if data == b'n':
      sock.sendto(s[i * 46080:(i + 1) * 46080], (UDP_IP, UDP_PORT))
  data, addr = sock.recvfrom(1)
  sock.sendto(b'eframe', (UDP_IP, UDP_PORT))

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

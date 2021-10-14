from djitellopy import tello
from time import time

me = tello.Tello()
me.connect()

print(me.get_battery())
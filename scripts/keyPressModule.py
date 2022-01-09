import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame

def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))

def getKey(keyname):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, "K_{}".format(keyname))
    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def main():
    if getKey("LEFT"):
        print("Left key pressed")
    if getKey("RIGHT"):
        print("Right key pressed")
    if getKey("UP"):
        print("UP")
    if getKey("DOWN"):
        print("DOWN")

if __name__ == "__main__":
    init()
    while True:
        main()
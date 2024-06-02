import pygame
import os
from mineNN import MineSweeperModel as MSM

class Game():

    def __init__(self,boardSize,screenSize):
        self.screenSize = screenSize
        self.boardSize = boardSize
        self.imageSize = screenSize[0] // boardSize[0], screenSize[1] // boardSize[1]
        self.images = self.loadImages()


    def run(self):
        pygame.init()
        running = True
        screen = pygame.display.set_mode(self.screenSize)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
    

                


    def loadImages(self):
        filenames = [f for f in os.listdir("images") if f.endswith('.png')]
        images = {}
        for name in filenames:
            imagename = os.path.splitext(name)[0] 
            curImage = pygame.image.load(os.path.join("images", name))
            curImage = pygame.transform.scale(curImage,self.imageSize)
            images[imagename] = curImage
        return images


    
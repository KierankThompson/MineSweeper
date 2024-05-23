import pygame
import os

class Game():

    def __init__(self,boardSize,screenSize):
        pygame.init()
        self.screenSize = screenSize
        self.boardSize = boardSize
        self.imageSize = screenSize[0] // boardSize[0], screenSize[1] // boardSize[1]


    def run(self):
        pygame.init()
        running = True
        screen = pygame.display.set_mode(self.screenSize)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.createBoard()
            pygame.display.flip()
    
    def createBoard(self):



    def loadImages(self):
        

    
import numpy as np
import torch
from torch import nn
import random

class MineSweeperModel():


    def __init__(self,boardSize):
        self.boardSizeX = boardSize[0]
        self.boardSizeY = boardSize[1]
        self.numSquares = boardSize[0] * boardSize[1]
        if self.boardSizeX == 9 and self.boardSizeY == 9:
            self.model = self.loadEasyModel()
        elif self.boardSizeX == 30 and self.boardSizeY == 16:
            self.model = self.loadHardModel()
        else:
            self.model = self.loadModel()
        
        
    
    def loadEasyModel(self):
        class EasyModel(nn.Module):
            def __init__(self):
                super(EasyModel, self).__init__()
                self.embedding = nn.Embedding(11, 9, max_norm=True)
                self.cnn = nn.Conv2d(in_channels = 9, out_channels = 16, kernel_size = 4,  stride = 1, bias=True)
                self.cnn2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size= 4, stride = 1, bias=True)
                self.lin = torch.nn.Linear(in_features = 32*3*3, out_features = 81, bias=True )
            def forward(self,input_tensor, embed = False):
                embedded = self.embedding(input_tensor)
                c1 = self.cnn(embedded)
                c2 = self.cnn2(c1)
                c2 = nn.Flatten()(c2)
                li = self.lin(c2)
                output = torch.nn.Sigmoid()(li)
                return output.view(-1, 9, 9)
        curModel = EasyModel()
        return curModel
    
    def loadHardModel(self):
        class ExpertModel(nn.Module):
            def __init__(self):
                super(ExpertModel, self).__init__()
                self.embedding = nn.Embedding(11, 9, max_norm=True)
                self.cnn = nn.Conv2d(in_channels = 9, out_channels = 16, kernel_size = 4,  stride = 1, bias=True)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride = 1, bias=True)
                self.lin = torch.nn.Linear(in_features = 32*24*10, out_features = 480, bias=True )
            def forward(self,input_tensor, embed = False):
                embedded = self.embedding(input_tensor)
                embedded = embedded.permute(0,3,1,2)
                c1 = self.cnn(embedded)
                c2 = self.conv2(c1)
                c2 = nn.Flatten()(c2)
                li = self.lin(c2)
                output = torch.nn.Sigmoid()(li)
                return output.view(-1,30, 16)
        curModel = ExpertModel()
        return curModel
    
    def loadModel(self):
        class MineModel(nn.Module):
            def __init__(self):
                super(MineModel, self).__init__()
                self.embedding = nn.Embedding(11, 9, max_norm=True)
                self.cnn = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=4, stride=1, padding=1, bias=True)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True) 

            def forward(self, input_tensor):
                embedded = self.embedding(input_tensor)
                embedded = embedded.permute(0, 3, 1, 2) 
                c1 = self.cnn(embedded)
                c2 = self.conv2(c1)
                c3 = self.conv3(c2)
                output = torch.sigmoid(c3)  
                return output.squeeze() 
        curModel = MineModel()
        return curModel


    def generateBoard(self,bombs,x,y):
        bombLocations = set()
        while len(bombLocations) < bombs:
            row = random.randint(0, x - 1)
            col = random.randint(0, y - 1)
            bombLocations.add((row,col))
        arr = [[9 for _ in range(y)] for _ in range(x)]
        for i in bombLocations:
            arr[i[0]][i[1]] = 10
        for j in range(len(arr)):
            for q in range(len(arr[j])):
                if arr[j][q] == 10:
                    continue
                bombCount = 0
                for dj in [-1, 0, 1]:
                    for dq in [-1, 0, 1]:
                        nj = j + dj
                        nq = q + dq
                        if 0 <= nj < len(arr) and 0 <= nq < len(arr[j]):
                            if arr[nj][nq] == 10:
                                bombCount += 1
                    
                arr[j][q] = bombCount
        return arr
    
    
    def generateData(self,numEntries):
        boards = []
        xBoards = []
        yBoards = []
        if self.boardSizeX == 9 and self.boardSizeY == 9:
            for _ in range(numEntries // 200):
                boards.append(self.generateBoard(10,9,9))
        elif self.boardSizeX == 30 and self.boardSizeY == 16:
            for _ in range(numEntries // 200):
                boards.append(self.generateBoard(10,9,9))
        else:
            for _ in range(numEntries // 200):
                boards.append(self.generateBoard(int(self.numSquares * 0.15),self.boardSizeX,self.boardSizeY))
        for board in boards:
            for _ in range(200):
                tempBoard = [[9 for _ in range(self.boardSizeY)] for _ in range(self.boardSizeX)]
                num_spots = random.randint(3, int(self.numSquares*0.8))
                for _ in num_spots():
                    row = random.randint(0, self.boardSizeX - 1)
                    col = random.randint(0, self.boardSizeY - 1)
                    while board[row][col] == 10:
                        row = random.randint(0, self.boardSizeX - 1)
                        col = random.randint(0, self.boardSizeY - 1)
                    tempBoard[row][col] = board[row][col]
                    def expandZeros(x,y):
                        if tempBoard[x][y] == 0:
                            for dj in [-1, 0, 1]:
                                for dq in [-1, 0, 1]:
                                    nj = x + dj
                                    nq = y + dq
                                    if 0 <= nj < self.boardSizeX and 0 <= nq < self.boardSizeY:
                                        tempBoard[nj][nq] = board[nj][nq]
                                        expandZeros(nj,nq)
                        return
                    expandZeros(row,col)
                xBoards.append(tempBoard)
                yBoards.append(board)
        return xBoards,yBoards

                




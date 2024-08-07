import copy
import numpy as np
import random
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageFont
import imageio
import time
import heapq


class EightTile():
    '''
    This class implements a basic 8-tile board when instantiated
    You can shuffle it using shuffle() to generate a puzzle
    After shuffling, you can use manual moves using ApplyMove()
    '''
    # class level variables for image and animation generation
    cellSize = 100  # cell within which a single char will be printed
    fontname = "calibri"
    fontsize = 100
    font = ImageFont.truetype(fontname, fontsize)  # font instance created
    simSteps = 10  # number of intermediate steps in each digit move

    @staticmethod
    def GenerateImage(b, BackColor=(0, 51, 102), ForeColor=(255, 253, 208)):
        """
        Generates an image given a board numpy array
        0s are simply neglected in the returned image

        if b is not a board object but a string, a single char image is generated
        """
        cellSize = EightTile.cellSize
        font = EightTile.font

        def draw_centered_text(draw, text, x, y, cellSize, font, ForeColor):
            # Calculate text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Calculate positions to center the text
            text_x = x + (cellSize - text_width) / 2
            text_y = y + (cellSize - text_height) / 2
            draw.text((text_x, text_y), text, font=font, fill=ForeColor)

        if isinstance(b, str):  # then a single character is expected, no checks
            img = Image.new('RGB', (cellSize, cellSize), BackColor)  # blank image
            imgPen = ImageDraw.Draw(img)  # pen to draw on the blank image
            draw_centered_text(imgPen, b, 0, 0, cellSize, font, ForeColor)  # write the char to img
        else:  # the whole board is to be processed
            img = Image.new('RGB', (3 * cellSize, 3 * cellSize), BackColor)  # blank image
            imgPen = ImageDraw.Draw(img)  # pen to draw on the blank image
            for row in range(3):  # go row by row
                y = row * cellSize
                for col in range(3):  # then columns
                    x = col * cellSize
                    txt = str(b[row, col]).replace('0', '')
                    # now that position of the current cell is fixed print into it
                    draw_centered_text(imgPen, txt, x, y, cellSize, font, ForeColor)  # write the character to board image
        # finally return whatever desired
        return np.array(img)  # return image as a numpy array

    def GenerateAnimation(board, actions, mName='puzzle', fps=15, debugON=False):
        # using each action collect images
        framez = []
        for action in actions:  # for every action generate animation frames
            frm = board.ApplyMove(action, True, debugON)
            EightTile.print_debug(f'frame:{len(frm)} for action {action}', debugON)
            framez += frm
        imageio.mimsave(mName + ".gif", framez, fps=fps)  # Creates gif out of list of images
        return framez

    def print_debug(mess, whether2print):
        # this is a simple conditional print,
        # you should prefer alternative methods for intensive printing
        if whether2print:
            print(mess)

    def __init__(me):
        # The eight tile board is a numpy array
        me.__board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        me.__winner = me.__board.copy()  # by default a winning board is given
        me.__x, me.__y = 2, 2  # keep track of the empty cell

    def shuffle(me, n=10, debugON=False):
        '''
        Randomly moves the empty tile, (i.e. the 0 element) around the gameboard
        n times and returns the moves from the initial to the last move
        Input:
            n: number of shuffles to perform, defaults to 10
        Output:
            a list of n entries [ [y1, x2], [y2, x2], ... , [yn, xn]]
            where [yi, xi] is the relative move of the empty tile at step i
            ex: [-1, 0] means that empty tile moved left horizontally
                note that:
                    on the corners there 2 moves
                    on the edges there are 3 moves
                    only at the center there are 4 moves

        Hence if you apply the negative of the returned list to the board
        step by step the puzzle should be solved!
        Check out ApplyMove()
        '''
        # depending on the current index possible moves are listed
        # think of alternative ways of achieving this, without using if conditions
        movez = [[0, 1], [-1, 0, 1], [-1, 0]]
        trace = []
        dxold, dyold = 0, 0  # to track past moves to avoid oscillations
        for i in range(n):
            # Note that move is along either x or y, but not both!
            # also no move at all is not accepted
            # we should also avoid the last state, i.e. an opposite move is not good
            dx, dy = 0, 0  # initial move is initialized to no move at all
            while (dx ** 2 + dy ** 2 != 1) or (
                    dx == -dxold and dy == -dyold):  # i.e. it is either no move or a diagonal move
                dx = random.choice(movez[me.__x])
                dy = random.choice(movez[me.__y])
            # now that we have the legal moves, we also have the new coordinates
            xn, yn = me.__x + dx, me.__y + dy  # record new coordinates
            trace.append([dy, dx])  # just keeping track of the move not the absolute position tomato, tomato
            me.__board[me.__y, me.__x], me.__board[yn, xn] = me.__board[yn, xn], me.__board[me.__y, me.__x]
            # enable print if debug is desired
            EightTile.print_debug(f'shuffle[{i}]: {me.__y},{me.__x} --> {yn},{xn}\n{me}\n', debugON)
            # finally update positions as well
            me.__x, me.__y = xn, yn

            dxold, dyold = dx, dy  # keeping track of old moves to avoid oscillations
        # finally return the sequence of shuffles
        return trace

    def ApplyMove(me, move, generateAnimation=False, debugON=False):
        '''
        Applies a single move to the board and updates it
        move is a list such that [deltaY, deltaX]
        this is manual usage, so it does not care about the previous moves
        if generateAnimation is set, a list of images will be returned that animates the move
        '''
        dy, dx = move[0], move[1]
        xn, yn = me.__x + dx, me.__y + dy  # record new coordinates
        imList = []
        if (dx ** 2 + dy ** 2 == 1 and 0 <= xn <= 2 and 0 <= yn <= 2):  # then valid
            if generateAnimation:
                # the value at the target will move to the current location
                cellSize = EightTile.cellSize
                simSteps = EightTile.simSteps
                c = me.__board[yn, xn]
                EightTile.print_debug(f'{c} is moving', debugON)
                # generate a template image
                temp = me.Board  # copy board
                temp[yn, xn] = 0  # blank the moving number as well which is kept in c
                tempimg = EightTile.GenerateImage(temp)  # get temp image
                tempnum = EightTile.GenerateImage(str(c))  # get the image for moving number
                # now at every animation step generate a new image
                # image will move in either along x or y, but linspace does not care
                xPos = np.linspace(xn * cellSize, me.__x * cellSize, simSteps + 1, dtype=int)
                yPos = np.linspace(yn * cellSize, me.__y * cellSize, simSteps + 1, dtype=int)
                Pos = np.vstack((yPos, xPos)).T  # position indices in target image are in rows
                EightTile.print_debug(f'{Pos}', debugON)
                # go over each pos pair to generate new images
                for p in range(Pos.shape[0]):
                    frm = tempimg.copy()  # generate a template image
                    xi, yi = Pos[p, 1], Pos[p, 0]
                    EightTile.print_debug(f'moving to {yi}:{xi}', debugON)
                    # '''
                    frm[yi:yi + me.cellSize, xi:xi + me.cellSize, :] = tempnum  # set image
                    EightTile.print_debug(f'frm = {frm.shape}, tempnum = {tempnum.shape}', debugON)
                    # finally add image to list
                    imList.append(frm)
            me.__board[me.__y, me.__x], me.__board[yn, xn] = me.__board[yn, xn], me.__board[me.__y, me.__x]
            me.__x, me.__y = xn, yn
            return imList
        else:
            return None

    def __str__(me):
        '''
        generates a printable version of the board
        note that * is the empty space and in the numpy representation of
        the board it is 0 (zero)
        '''
        return tabulate([[str(x).replace('0', '*') for x in c] for c in np.ndarray.tolist(me.__board)], tablefmt="grid",
                        stralign="center")

    @property
    def Position(me):
        # returns the position of the empty cell
        return [me.__y, me.__x]

    @property
    def Board(me):
        # returns a numpy array stating the current state of the board
        return me.__board.copy()  # return a copy of the numpy array

    @property
    def isWinner(me):
        # returns true, if current board is a winner
        return np.array_equal(me.__winner, me.__board)

    @property
    def BoardImage(me):
        return EightTile.GenerateImage(me.Board)


class BoardStates:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move  # the move applied to parent node to reach this current node
        self.blank_pos = board.Position
        self.all_moves = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])  # all possible moves of any tile
        if self.parent:
            self.g = parent.g + 1  # g parameter calculated as number of moves to reach current board
        else:
            self.g = 0

    def new_boards(self):
        """
        This method returns a list of all possible new positions for the current blank tile position
        """
        moves = [m for m in self.all_moves if all(0 <= x <= 2 for x in np.add(m, self.blank_pos))]
        children = []
        for move in moves:
            # generates all possible next board layouts as an EightTile instance
            child = copy.deepcopy(self.board)
            child.ApplyMove(move)
            children.append(child)
        return children, moves

    @property
    # Calculation of the heuristic parameter (distance from goal). Manhattan distance is used.
    def h(self):
        distance = 0
        for row in range(3):
            for column in range(3):
                if self.board.Board[row][column] != 0:
                    x, y = divmod(self.board.Board[row][column] - 1, 3)
                    distance += abs(x - row) + abs(y - column)
        return distance

    @property
    # f function (sum of cost function and heuristic value)
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f


class Solve8:
    def __init__(self):
        self.open = []
        self.seen = set()

    def __str__(self):
        return 'The Blind Daters'

    def array_in_set(self, arr):
        # function to check if a numpy array is in a list of numpy arrays (couldn't find a better method)
        return tuple(map(tuple, arr)) in self.seen

    def Solve(self, tile):
        """
        Input: an 8 tile object
        Output: a list of moves which will generate the winning boar
        when applied one after the other to the input 8 tile object
        check out the example moves above, howerever they are reversed and
        multiplied by -1, in your case no reverse or -1, just apply them
        one after the other
        movez should contain the minimum number of moves needed to solve the puzzle
        """

        current = BoardStates(tile)
        heapq.heappush(self.open, (current.f, current))

        while self.open:
            _, node = heapq.heappop(self.open)
            if node.board.isWinner:
                movez = []
                while node.parent:
                    movez.append(node.move)
                    node = node.parent
                return list(reversed(movez))
            self.seen.add(tuple(map(tuple, node.board.Board)))
            childs, mover = node.new_boards()
            for c, kid in enumerate(childs):
                current = BoardStates(kid, node, mover[c])
                if not self.array_in_set(current.board.Board):
                    heapq.heappush(self.open, (current.f, current))
        return []

# example usage
t = EightTile()

# Shuffle the puzzle (replace 81 with any number of shuffles)
t.shuffle(81)

# Initial board
initial_board_img = Image.fromarray(t.BoardImage)
initial_board_img.save('initial_board.png')

# Solve the puzzle and get the sequence of moves
start_time = time.time()
p = Solve8()
solution_moves = p.Solve(t)
print(len(solution_moves))
time_duration = time.time() - start_time
print(time_duration)

# Solve the board and generate animation
frames = EightTile.GenerateAnimation(t, solution_moves, mName='puzzle_solution')

# Solved Board
solved_board_img = Image.fromarray(t.BoardImage)
solved_board_img.save('solved_board.png')

# Save the animation as a GIF file
imageio.mimsave('puzzle_solution.gif', frames, fps=15)
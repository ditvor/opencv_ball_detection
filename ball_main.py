import pygame
import sys
import cv2 as cv
import numpy as np
import math
import random

# define colour scheme
sky_blue = (137, 209, 254)
dark_green = (0, 71, 49)
crimson = (223, 31, 61)
gray = (113, 111, 111)

# define screen and objects dimensions
screen_width = 1024
screen_height = 640
size = [screen_width, screen_height]
ball_size = 25

# ball starting point

ball_start_x = 100
ball_start_y = 320

# define computer vision region
cv_left = 300
cv_top = 0
cv_width = 400
cv_height = screen_height

# initial parameters
gravity = 0.045


class Ball:
    """ Creates a ball with initial parameters. """
    def __init__(self):
        # coordinates
        self.x = 0
        self.y = 0

        # components of velocity vector and its angle(alpha)
        self.velocity_0_x = 0
        self.velocity_0_y = 0
        self.alpha = 0

        # each ball moves in its own time frame
        self.time = 0

    def create_trajectory(self):
        """ Creates a path for the ball in a gravitational field. """
        self.time += 1
        self.x = ball_start_x + self.velocity_0_x * self.time * math.cos(self.alpha)
        self.y = ball_start_y - self.velocity_0_y * self.time * math.sin(self.alpha) + (gravity * self.time ** 2) / 2


class Paddle:
    """ Introduces a paddle to the game. """
    def __init__(self):
        # coordinates of left upper corner
        self.x = 0
        self.y = 0

        # dimensions
        self.width = 0
        self.height = 0


class DetectObject:

    def __init__(self, frame):
        self.frame = frame

    def screen_pygame_to_opencv(self):
        """ Converts a pygame surface into opencv image."""

        frame_cv = pygame.surfarray.array3d(self.frame)
        frame_cv = cv.transpose(frame_cv)
        frame_cv = cv.cvtColor(frame_cv, cv.COLOR_RGB2BGR)

        return frame_cv

    def detect_ball(self):
        """ Recognizes a ball on a frame.
            Returns coordinates and radius as np.array.
        """

        # creates grayish image of a surface
        frame_gray = cv.cvtColor(self.screen_pygame_to_opencv(), cv.COLOR_BGR2GRAY)

        # run circle recognition
        circle = cv.HoughCircles(frame_gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=15)

        if circle is not None:
            # convert the (x, y) coordinates and radius of the circle to integers
            circle = np.round(circle[0, :]).astype("int")

        return circle


def make_ball():
    """ Function to create a new ball. """
    ball = Ball()

    # starting position of the ball
    ball.x = ball_start_x
    ball.y = ball_start_y

    # components of velocity vector
    ball.velocity_0_x = 5 + random.randrange(1, 10)
    ball.velocity_0_y = 3 + random.randrange(1, 5)

    # angle of the velocity vector
    ball.alpha = math.atan(ball.velocity_0_y / ball.velocity_0_x)

    return ball


def make_paddle():
    """ Function that adds a paddle to the screen. """
    paddle = Paddle()

    # starting position of the paddle
    paddle.x = 1000
    paddle.y = 270

    # dimensions of the paddle
    paddle.width = 15
    paddle.height = 100

    return paddle


def make_field_of_vision(screen):
    """ Sets a region on the initial surface for the computer vision. """
    field_of_vision = pygame.Rect(cv_left, cv_top, cv_width, cv_height)
    field_of_vision = screen.subsurface(field_of_vision)

    return field_of_vision


def predict_landing(paddle_axis, positions_list):
    """ Predicts a landing point on the paddle axis. """
    points = np.array(positions_list)

    # extract values from numpy array
    coord_x = points[:, 0]
    coord_y = points[:, 1]

    fitting = np.polyfit(coord_x, coord_y, 2)
    predict_func = np.poly1d(fitting)

    # prediction of ball landing on the paddle axis
    landing_point = int(predict_func(paddle_axis))

    return landing_point


def main():
    """ Creates a canvas and starts the game. """
    pygame.init()

    # set the screen dimensions
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Ball shooting machine")

    # loop the game while the window is not closed
    finished = False

    # manage how fast the screen updates
    clock = pygame.time.Clock()

    ball = make_ball()
    paddle = make_paddle()

    # saves coordinates of a detected ball
    # will be used for the prediction of the ball landing
    position_ball_list = []

    while not finished:
        # event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                # hit space bar to restart the game
                if event.key == pygame.K_SPACE:
                    main()

        ball.create_trajectory()

        # set the screen background
        screen.fill(sky_blue)

        # draw objects on the screen
        pygame.draw.line(screen, gray, [cv_left, cv_top], [cv_left, cv_height], 3)
        pygame.draw.line(screen, gray, [cv_left+cv_width, cv_top], [cv_left+cv_width, cv_height], 3)
        pygame.draw.circle(screen, crimson, [ball.x, ball.y], ball_size)
        pygame.draw.rect(screen, dark_green, [paddle.x, paddle.y, paddle.width, paddle.height])

        # prepare a surface for opencv
        # try to detect a ball on the surface
        field_of_vision = make_field_of_vision(screen)
        frame_with_object = DetectObject(field_of_vision)
        detected_ball = frame_with_object.detect_ball()

        # if ball is detected put its coordinates in the list
        if detected_ball is not None:
            for (x, y, r) in detected_ball:
                position_ball_list.append((cv_left + x, y))

        # if we have enough data - let's predict the landing point
        # move the paddle to catch the ball
        if len(position_ball_list) > 30:
            predicted_landing_position = predict_landing(paddle.x, position_ball_list)
            paddle.y = predicted_landing_position - paddle.height / 2

        # limit to 30 FPS
        clock.tick(30)

        # update the screen with all the figures
        pygame.display.flip()

        # close everything
    pygame.quit()


if __name__ == "__main__":
    main()

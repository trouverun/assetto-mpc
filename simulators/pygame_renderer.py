import pygame
import numpy as np
import matplotlib as plt
import config
from utils.maths import rotate_around
from scipy import interpolate


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
DEBUG = (0, 255, 255)


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode([config.map_resolution, config.map_resolution])
        self.car_image = pygame.image.load('simulators/car_graphic.png')
        scaler = config.map_resolution / config.map_distance
        self.car_px_width = scaler*1.894 * 2
        self.car_px_height = scaler*4.894 * 2
        self.car_image = pygame.transform.scale(self.car_image, (self.car_px_width, self.car_px_height))

    def render_scenario(self, car_pos_hdg, track_centerline, position_history, position_vel_horizon, tightened_track_left, tightened_track_right):
        for _ in pygame.event.get():
            pass

        surf = pygame.Surface((config.map_resolution, config.map_resolution))
        surf.fill(WHITE)

        car_pos = car_pos_hdg[:2]
        car_hdg = car_pos_hdg[2]
        track_centerline = track_centerline[np.sqrt(np.square(track_centerline[:, 1:3] - car_pos).sum(axis=1)) < 3*config.map_distance]

        track_centerline[:, 1:3] = rotate_around(car_pos, car_hdg, track_centerline[:, 1:3])
        position_history[:, :2] = rotate_around(car_pos, car_hdg, position_history[:, :2])
        if position_vel_horizon is not None:
            position_vel_horizon[:, :2] = rotate_around(car_pos, car_hdg, position_vel_horizon[:, :2])
        tightened_track_left[:, :2] = rotate_around(car_pos, car_hdg, tightened_track_left[:, :2])
        tightened_track_right[:, :2] = rotate_around(car_pos, car_hdg, tightened_track_right[:, :2])

        p1x = int(config.map_resolution / 2)
        scaler_x = (config.map_resolution / 2) / config.map_distance
        p1y = int(config.map_resolution / 2)
        scaler_y = (config.map_resolution / 2) / config.map_distance

        cx_spline = interpolate.splrep(track_centerline[:, 0], track_centerline[:, 1], k=3)
        cy_spline = interpolate.splrep(track_centerline[:, 0], track_centerline[:, 2], k=3)
        dx = interpolate.splev(track_centerline[:, 0], cx_spline, der=1)
        dy = interpolate.splev(track_centerline[:, 0], cy_spline, der=1)
        angles = np.arctan2(dy, dx)
        last_center, last_left, last_right = None, None, None
        for midpoint, angle in zip(track_centerline, angles):
            x = midpoint[1]
            y = midpoint[2]
            w = midpoint[3]

            center = [p1x + int(x * scaler_x), p1y - int(y * scaler_y)]
            left = [
                p1x + int((x + w*np.cos(angle-np.deg2rad(90))) * scaler_x),
                p1y - int((y + w*np.sin(angle-np.deg2rad(90))) * scaler_y)
            ]
            right = [
                p1x + int((x + w*np.cos(angle+np.deg2rad(90))) * scaler_x),
                p1y - int((y + w*np.sin(angle+np.deg2rad(90))) * scaler_y)
            ]

            if last_center is not None:
                pygame.draw.line(surf, BLACK,
                                 last_center,
                                 center,
                                 2)
                pygame.draw.line(surf, BLACK,
                                 last_left,
                                 left,
                                 2)
                pygame.draw.line(surf, BLACK,
                                 last_right,
                                 right,
                                 2)
            last_center = center
            last_left = left
            last_right = right

        last_left, last_right = None, None
        for left, right in zip(tightened_track_left, tightened_track_right):
            left = [
                p1x + int(left[0] * scaler_x),
                p1y - int(left[1] * scaler_y)
            ]
            right = [
                p1x + int(right[0] * scaler_x),
                p1y - int(right[1] * scaler_y)
            ]

            if last_left is not None:
                pygame.draw.line(surf, RED,
                                 last_left,
                                 left,
                                 2)
                pygame.draw.line(surf, RED,
                                 last_right,
                                 right,
                                 2)
            last_left = left
            last_right = right

        cmap = plt.colormaps["cool"]
        for pt in position_history:
            color = tuple([int(255 * c) for c in cmap(pt[2] / (250*config.KM_H))[:3]])
            pygame.draw.circle(surf, color, [p1x + int(pt[0] * scaler_x), p1y - int(pt[1] * scaler_y)], 5)
        if position_vel_horizon is not None:
            for pt in position_vel_horizon:
                color = tuple([int(255 * c) for c in cmap(pt[2] / (250*config.KM_H))[:3]])
                pygame.draw.circle(surf, color, [p1x + int(pt[0] * scaler_x), p1y - int(pt[1] * scaler_y)], 5)

        pygame.draw.circle(surf, BLACK, [p1x, p1y], 5)

        self.screen.blits([
            (surf, (0, 0)),
            #(self.car_image, (p1x - self.car_px_width/2, p1y - self.car_px_height/2)),
        ])
        pygame.display.flip()
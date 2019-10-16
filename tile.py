import numpy as np
from scipy.spatial.transform import Rotation
import imageio
import api
import matplotlib.pyplot as plt
from pprint import pprint

TILE_SIZE = 256
WGS84_RADIUS = 6378137  # Meters
WGS84_CIRCUMFERENCE = 2 * np.pi * WGS84_RADIUS
TILE_RESOLUTION = WGS84_CIRCUMFERENCE / TILE_SIZE
ORIGIN_METERS = WGS84_CIRCUMFERENCE / 2


class TileMap:
    METERS_X = 0
    METERS_Y = 1
    METERS_Z = 2
    DATA_DEPTH = 3

    def __init__(self, corner1, corner2, zoom):
        self.zoom = zoom
        self._tiles = self.corners_to_tiles(corner1, corner2, zoom)
        num_y_pixels = len(self._tiles) * TILE_SIZE
        num_x_pixels = len(self._tiles[0]) * TILE_SIZE
        self._data = np.zeros((self.DATA_DEPTH, num_y_pixels, num_x_pixels))

        # Get all the data
        self.download_tiles()

        # Recalculate pixel space
        ox, oy, _ = self._tiles[0][0]  # Origin
        ox, oy = self.tile_coords_to_pixel(ox, oy)
        px = np.arange(ox, ox + num_x_pixels)
        py = np.arange(oy, oy + num_y_pixels)

        # Back out to meters (Mercator)
        mx, my = self.pixel_to_mercator(px, py, self.zoom)

        # Convert to lat/lon bounds
        self._lat, self._lon = self.mercator_to_latlon(mx, my)

        # Trim data
        # self.trim(corner1, corner2)

        # Un-mercatorize
        self.reproject()

        # Set x, y min to 0, 0
        self.zeroize_xy()

    def trim(self, corner1, corner2):
        lat_bounds = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
        lon_bounds = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
        lat_within_bounds = np.logical_and(
            self._lat >= lat_bounds[0],
            self._lat <= lat_bounds[1],
        )
        lon_within_bounds = np.logical_and(
            self._lon >= lon_bounds[0],
            self._lon <= lon_bounds[1],
        )

        mask = np.ix_(np.arange(self.DATA_DEPTH), lat_within_bounds, lon_within_bounds)
        self._lat = self._lat[lat_within_bounds]
        self._lon = self._lon[lon_within_bounds]
        self._data = self._data[mask]

    def reproject(self):
        latlon = np.meshgrid(self._lat, self._lon, indexing='ij')
        # Create a x, y, z point cloud with 0 elevation. This elevation will be added to the real data. This lets us
        # handle the fact that the edges of the cloud will be "below sea level" intelligently
        xyz = np.stack(self.geodetic_to_ecef(*latlon, np.zeros(self.elevation.shape)))
        original_shape = xyz.shape
        xyz = xyz.reshape(3, -1).T  # Reshape to list of 3d points

        # Rotate in longitude (about the z-axis) first, to align the center of the lat/lon region with longitude 0
        # Then, rotate in latitude (about the y-axis), to align the center of the region with the north pole
        middle_lat = np.median(self._lat)
        middle_lon = np.median(self._lon)
        R = Rotation.from_euler('zy', (360 - middle_lon, -(90 - middle_lat)), degrees=True)
        xyz = R.apply(xyz)

        xyz = xyz.T.reshape(original_shape)
        xyz[2, :, :] -= WGS84_RADIUS
        xyz[2, :, :] -= xyz[2, :, :].min()  # Get offset to 0

        self._data += xyz

    # TODO: Use WGS84 spheroid instead of sphere
    # TODO: This requires the rotation/reproject step also consider ellipsoidal coordinates
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    @staticmethod
    def geodetic_to_ecef(lat, lon, h):
        rlat, rlon = np.radians(lat), np.radians(lon)
        x = (WGS84_RADIUS + h) * np.cos(rlat) * np.cos(rlon)
        y = (WGS84_RADIUS + h) * np.cos(rlat) * np.sin(rlon)
        z = (WGS84_RADIUS + h) * np.sin(rlat)
        return x, y, z

    def zeroize_xy(self):
        self._data[self.METERS_X, :, :] -= self._data[self.METERS_X, :, :].min()
        self._data[self.METERS_Y, :, :] -= self._data[self.METERS_Y, :, :].min()

    def scale(self, scale_factor):
        self._data *= scale_factor

    @property
    def elevation(self):
        return self._data[self.METERS_Z, :, :]

    @property
    def xyz(self):
        return self._data[self.METERS_X:self.METERS_Z + 1, :, :]

    # TODO: Make this take an arbitrary polygon and generate all tiles the polygon overlaps
    @classmethod
    def corners_to_tiles(cls, corner1, corner2, zoom):
        # Convert corners to tile coordinates
        mercator1 = cls.latlon_to_mercator(*corner1)
        pixel1 = cls.mercator_to_pixel(*mercator1, zoom)
        tile_coords1 = cls.pixel_to_tile_coords(*pixel1)
        tile_coords1 = [int(_) for _ in tile_coords1]

        mercator2 = cls.latlon_to_mercator(*corner2)
        pixel2 = cls.mercator_to_pixel(*mercator2, zoom)
        tile_coords2 = cls.pixel_to_tile_coords(*pixel2)
        tile_coords2 = [int(_) for _ in tile_coords2]

        lo_x = min(tile_coords1[0], tile_coords2[0])
        hi_x = max(tile_coords1[0], tile_coords2[0])
        lo_y = min(tile_coords1[1], tile_coords2[1])
        hi_y = max(tile_coords1[1], tile_coords2[1])

        return [[(x, y, zoom) for x in range(lo_x, hi_x + 1)] for y in range(lo_y, hi_y + 1)]

    @staticmethod
    def resolution(zoom):
        return TILE_RESOLUTION / (2 ** zoom)

    @staticmethod
    def latlon_to_mercator(lat, lon):
        mx = WGS84_RADIUS * np.radians(lon)
        my = WGS84_RADIUS * -np.log(np.tan((np.pi / 4) + (np.radians(lat) / 2)))
        return mx, my

    @staticmethod
    def mercator_to_latlon(mx, my):
        lon = np.degrees(mx / WGS84_RADIUS)
        lat = -np.degrees(2 * np.arctan(np.exp(my/WGS84_RADIUS)) - (np.pi / 2))
        return lat, lon

    @classmethod
    def mercator_to_pixel(cls, mx, my, zoom):
        r = cls.resolution(zoom)
        px = (mx + ORIGIN_METERS) / r
        py = (my + ORIGIN_METERS) / r
        return px, py

    @classmethod
    def pixel_to_mercator(cls, px, py, zoom):
        r = cls.resolution(zoom)
        mx = (px * r) - ORIGIN_METERS
        my = (py * r) - ORIGIN_METERS
        return mx, my

    @staticmethod
    def pixel_to_tile_coords(px, py):
        tx = (px / TILE_SIZE)
        ty = (py / TILE_SIZE)
        return tx, ty

    @staticmethod
    def tile_coords_to_pixel(tx, ty):
        px = tx * TILE_SIZE
        py = ty * TILE_SIZE
        return px, py

    @staticmethod
    def rgb_to_meters(rgb):
        return (rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256) - 32768

    def download_tiles(self):
        for i, row in enumerate(self._tiles):
            for j, tile in enumerate(row):
                filename = api.save_tile(*tile)
                arr = np.array(imageio.imread(filename))
                height = self.rgb_to_meters(arr)
                ii = np.s_[i * TILE_SIZE:i * TILE_SIZE + TILE_SIZE]
                jj = np.s_[j * TILE_SIZE:j * TILE_SIZE + TILE_SIZE]
                self._data[self.METERS_Z, ii, jj] = height

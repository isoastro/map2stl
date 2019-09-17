import numpy as np
import os.path
import urllib.request
import imageio
import matplotlib.pyplot as plt
from collections import namedtuple
from dataclasses import dataclass

TILE_SIZE = 256
WGS84_RADIUS = 6378137  # Meters
WGS84_CIRCUMFERENCE = 2 * np.pi * WGS84_RADIUS
TILE_RESOLUTION = WGS84_CIRCUMFERENCE / TILE_SIZE
ORIGIN_METERS = WGS84_CIRCUMFERENCE / 2


def _resolution(zoom):
    """Calculate meters/pixel (at equator)"""
    return TILE_RESOLUTION / (2 ** zoom)


class LatLon(namedtuple('LatLon', 'lat lon')):
    def mercator(self):
        """Convert latitude and longitude (degrees) in WGS84 to meters in Spherical Mercator"""
        # https://en.wikipedia.org/wiki/Mercator_projection#Derivation_of_the_Mercator_projection

        x = WGS84_RADIUS * (np.radians(self.lon))

        lat = np.radians(self.lat)
        y = WGS84_RADIUS * np.log(np.tan((np.pi / 4) + (lat / 2)))

        return Mercator(x, y)

    def pixel(self, zoom):
        return self.mercator().pixel(zoom)

    def tile(self, zoom):
        return self.pixel(zoom).tile()


class Mercator(namedtuple('Mercator', 'x y')):
    def lat_lon(self):
        """Convert meters in Spherical Mercator to latitude and longitude (degrees) in WGS84"""
        # https://en.wikipedia.org/wiki/Mercator_projection#Derivation_of_the_Mercator_projection

        lon = np.degrees(self.x / WGS84_RADIUS)

        lat = 2 * np.arctan(np.exp(self.y / WGS84_RADIUS)) - (np.pi / 2)
        lat = np.degrees(lat)

        return LatLon(lat, lon)

    def pixel(self, zoom):
        """Convert meters in Spherical Mercator to a pixel location given a zoom level"""
        r = _resolution(zoom)
        x = (self.x + ORIGIN_METERS) / r
        y = (self.y + ORIGIN_METERS) / r

        return Pixel(x, y, zoom)

    def tile(self, zoom):
        return self.pixel(zoom).tile()


class Pixel(namedtuple('Pixel', 'x y zoom')):
    @property
    def resolution(self):
        """Calculate meters/pixel"""
        return _resolution(self.zoom)

    def mercator(self):
        """Convert a pixel location to meters in Spherical Mercator"""
        r = self.resolution

        x = self.x * r - ORIGIN_METERS
        y = self.y * r - ORIGIN_METERS

        return Mercator(x, y)

    def tile(self):
        """Convert a pixel location to a tile x, y, and zoom. Zoom is not needed for calculation but needed for the tile
            to be useful"""
        x = (self.x / TILE_SIZE) - 1
        y = (self.y / TILE_SIZE) - 1

        return TileCoords(x, y, pixel.zoom)

    def lat_lon(self):
        return self.mercator().lat_lon()


class TileCoords(namedtuple('Tile', 'x y zoom')):
    @property
    def resolution(self):
        """Calculate meters/pixel"""
        return _resolution(self.zoom)

    def pixel(self):
        """Convert a tile x, y, and zoom to a pixel location. Zoom is carried through to make the pixel meaningful"""
        x = (self.x + 1) * TILE_SIZE
        y = (self.y + 1) * TILE_SIZE

        return Pixel(x, y, self.zoom)

    def mercator(self):
        return self.pixel().mercator()

    def lat_lon(self):
        return self.mercator().lat_lon()


class Tile(TileCoords):
    _data = None

    # TODO: This is kind of dumb
    @property
    def data(self):
        if self._data is None:
            x = np.zeros((TILE_SIZE, TILE_SIZE))
            y = np.zeros((TILE_SIZE, TILE_SIZE))
            origin = self.pixel()
            origin_m = origin.mercator()
            for i in range(TILE_SIZE):
                for j in range(TILE_SIZE):
                    pixel = Pixel(origin.x + i, origin.y + j, origin.zoom)
                    merc = pixel.mercator()
                    x[i, j] = merc.x
                    y[i, j] = merc.y
            self._data = np.dstack((x - origin_m.x, y - origin_m.y))
        return self._data


def rgb_to_meters(arr):
    """Convert an RGB array (image) to a 2d array of elevation values"""
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return (r * 256 + g + b / 256) - 32768



def get_grid_from_file(filename):
    arr = np.array(imageio.imread(filename))
    x = np.arange(TILE_SIZE)
    X, Y = np.meshgrid(x, x)
    Z = rgb_to_meters(arr)
    return np.dstack((X, Y, Z))


if __name__ == '__main__':
    # lat, lon = 46.852248, -121.757702  # Rainier
    lat = 85
    lon = 54
    latlon = LatLon(lat, lon)
    print('Original lat and lon')
    print(f'lat == {latlon.lat}')
    print(f'lon == {latlon.lon}')
    print()

    merc = latlon.mercator()
    print('Converted to mercator')
    print(f'x == {merc.x}')
    print(f'y == {merc.y}')
    print()

    zoom = 11
    pixel = merc.pixel(zoom)
    print(f'Converted to pixel (zoom == {zoom})')
    print(f'x == {pixel.x}')
    print(f'y == {pixel.y}')
    print(f'zoom == {pixel.zoom}')
    print()

    tile = pixel.tile()
    print('Converted to tile')
    print(f'x == {tile.x}')
    print(f'y == {tile.y}')
    print(f'zoom == {tile.zoom}')
    print()

    print('Combinations')
    print()

    methods = ['lat_lon', 'mercator', 'pixel', 'tile']
    for obj in (latlon, merc, pixel, tile):
        for meth in methods:
            fn = None
            try:
                fn = getattr(obj, meth)
            except AttributeError:
                continue

            try:
                res = fn()
                print(f'{obj}.{meth}() = {res}')
            except TypeError:
                res = fn(zoom)
                print(f'{obj}.{meth}(zoom={zoom}) = {res}')

    print()
    print('Converting tile coordinates to meters')

    tile = TileCoords(1330, 2043, 11)
    origin = tile.pixel()

    t = Tile(*tile)
    print(t.data)

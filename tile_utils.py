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

        x = WGS84_RADIUS * np.radians(self.lon)

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

        return TileCoords(x, y, self.zoom)

    def lat_lon(self):
        return self.mercator().lat_lon()


class TileCoords(namedtuple('Tile', 'x y zoom')):
    @property
    def resolution(self):
        """Calculate meters/pixel"""
        return _resolution(self.zoom)

    def asint(self):
        return self.__class__(*map(int, map(round, self)))

    def pixel(self):
        """Convert a tile x, y, and zoom to a pixel location. Zoom is carried through to make the pixel meaningful"""
        x = (self.x + 1) * TILE_SIZE
        y = (self.y + 1) * TILE_SIZE

        return Pixel(x, y, self.zoom)

    def mercator(self):
        return self.pixel().mercator()

    def lat_lon(self):
        return self.mercator().lat_lon()


def rgb_to_meters(arr):
    """Convert an RGB array (image) to a 2d array of elevation values"""
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return (r * 256 + g + b / 256) - 32768


def get_grid_from_file(filename):
    arr = np.array(imageio.imread(filename))
    return rgb_to_meters(arr)


def corners_to_tiles(corner1, corner2, zoom):
    """Convert a bounding rectangle in latitude and longitude into a 2d tuple of tile coordinates"""
    xy1 = corner1.tile(zoom).asint()
    xy2 = corner2.tile(zoom).asint()

    lo_x = min(xy1.x, xy2.x)
    hi_x = max(xy1.x, xy2.x)
    lo_y = min(xy1.y, xy2.y)
    hi_y = max(xy1.y, xy2.y)

    return tuple(tuple(TileCoords(x, y, zoom) for x in range(lo_x, hi_x + 1)) for y in range(lo_y, hi_y + 1))


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

    nw = LatLon(46.931359, -121.898533)
    se = LatLon(46.761857, -121.624521)
    tiles = corners_to_tiles(nw, se, 11)
    print(tiles)

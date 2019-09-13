import numpy as np
import os.path
import urllib.request
import imageio
import matplotlib.pyplot as plt
from collections import namedtuple

TILE_SIZE = 256
WGS84_RADIUS = 6378137  # Meters
WGS84_CIRCUMFERENCE = 2 * np.pi * WGS84_RADIUS
TILE_RESOLUTION = WGS84_CIRCUMFERENCE / TILE_SIZE
ORIGIN_METERS = WGS84_CIRCUMFERENCE / 2


def _resolution(zoom):
    """Calculate meters/pixel"""
    return TILE_RESOLUTION / (2 ** zoom)


class LatLon(namedtuple('LatLon', 'lat lon')):
    def mercator(self):
        """Convert latitude and longitude (degrees) in WGS84 to meters in Spherical Mercator"""
        # https://en.wikipedia.org/wiki/Mercator_projection#Derivation_of_the_Mercator_projection

        x = WGS84_RADIUS * (np.radians(self.lon))

        lat = np.radians(self.lat)
        y = WGS84_RADIUS * np.log(np.tan((np.pi / 4) + (lat / 2)))

        return Mercator(x, y)

    def pixel(self):
        return self.mercator().pixel()

    def tile(self):
        return self.pixel().tile()


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

    def tile(self):
        return self.pixel().tile()


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

        return Tile(x, y, pixel.zoom)

    def lat_lon(self):
        return self.mercator().lat_lon()


class Tile(namedtuple('Tile', 'x y zoom')):
    @property
    def resolution(self):
        """Calculate meters/pixel"""
        return _resolution(self.zoom)

    def pixel(self):
        """Convert a tile x, y, and zoom to a pixel location. Zoom is carried through to make the pixel meaningful"""
        x = (tile.x + 1) * TILE_SIZE
        y = (tile.y + 1) * TILE_SIZE

        return Pixel(x, y, tile.zoom)

    def mercator(self):
        return self.pixel().mercator()

    def lat_lon(self):
        return self.mercator().lat_lon()


def sec(x):
    return 1.0 / np.cos(x)


def rgb_to_meters(arr):
    """Convert an RGB array (image) to a 2d array of elevation values"""
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return (r * 256 + g + b / 256) - 32768


def int_and_dec(x):
    """Returns integer and decimal part of a number"""
    i = int(x)
    return i, float(x - i)


def lat_lon_to_tile(lat, lon, zoom):
    """Converts a given latitude and longitude (degrees) to a tile x, y"""
    n = 2.0 ** zoom
    x = n * ((lon + 180.0) / 360.0)
    lat = np.radians(lat)
    y = n * (1 - (np.log(np.tan(lat) + sec(lat)) / np.pi)) / 2.0
    return x, y


def pixels_to_meters(x, y, zoom):
    res = 2 * np.pi * 6378137 / TILE_SIZE
    res /= (2 ** zoom)
    origin = 2 * np.pi * 6378137 / 2.0

    mx = x * res - origin
    my = y * res - origin


def tile_to_lat_lon(x, y, zoom):
    """Converts a given tile x, y to latitude and longitude (degrees)"""
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_to_url(x, y, zoom):
    url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom:d}/{x:d}/{y:d}.png'
    return url


def tile_to_filename(x, y, zoom, output_dir='tiles'):
    filename = os.path.join(output_dir, f'{zoom:d}_{x:d}_{y:d}.png')
    return filename


def save_tile(x, y, zoom):
    url = tile_to_url(x, y, zoom)
    filename = tile_to_filename(x, y, zoom)
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f'Saved {url} to {filename}')
    return filename


def get_grid_from_file(filename):
    arr = np.array(imageio.imread(filename))
    x = np.arange(TILE_SIZE)
    X, Y = np.meshgrid(x, x)
    Z = rgb_to_meters(arr)
    return np.dstack((X, Y, Z))


def build_grid(lat, lon, zoom):
    x, y = lat_lon_to_tile(lat, lon, zoom)
    x = np.sort(np.unique(x.astype(np.int)))
    y = np.sort(np.unique(y.astype(np.int)))
    print(x, y)
    grid = np.zeros((len(y) * TILE_SIZE, len(x) * TILE_SIZE, 3))
    for i, yidx in enumerate(y):
        for j, xidx in enumerate(x):
            filename = save_tile(xidx, yidx, zoom)
            subgrid = get_grid_from_file(filename)
            subgrid[:, :, 0] += j * TILE_SIZE
            subgrid[:, :, 1] += i * TILE_SIZE
            print(subgrid.shape)
    return grid


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

    pixel2 = tile.pixel()
    print('Converted back to pixel')
    print(f'x == {pixel2.x}')
    print(f'y == {pixel2.y}')
    print(f'zoom == {pixel2.zoom}')
    print()

    merc2 = pixel2.mercator()
    print(f'Converted back to mercator')
    print(f'x == {merc2.x}')
    print(f'y == {merc2.y}')
    print()

    latlon2 = merc2.lat_lon()
    print(f'Converted back to lat and lon')
    print(f'lat == {latlon2.lat}')
    print(f'lon == {latlon2.lon}')

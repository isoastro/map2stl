import numpy as np
from math import radians, log, tan, cos, pi, floor, ceil
import urllib.request
import os
import imageio
import struct

TILE_SIZE = 256

# TODO: Generate STL file from 2d elevation data
# TODO: Also generate x, y data to account for mercator - this probably doesn't matter a ton, but it could be fun
# TODO: Make easier to use (i.e., incorporate zoom as parameter)


# Because this doesn't exist otherwise...
def sec(x):
    return 1 / cos(x)


def int_and_dec(x):
    """Returns integer and decimal part of a number"""
    i = int(x)
    return i, float(x - i)


def resolution(zoom, latitude):
    """Resolution of a single pixel in meters/pixel"""
    res = 40075.016686 * 1000 / TILE_SIZE  # 40075.016686 km equator length in WGS-84
    return res * cos(radians(latitude)) / (2.0 ** zoom)


ZOOM = 13
url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{{:d}}/{{:d}}.png'
output_dir = 'tiles'
fname_template = os.path.join(output_dir, f'{ZOOM}_{{:d}}_{{:d}}.png')


def lat_lon_to_tile(lat, lon):
    """Converts a given latitude and longitude (degrees) to a tile x, y"""
    n = 2 ** ZOOM
    x = n * ((lon + 180.0) / 360.0)
    lat = radians(lat)
    y = n * (1 - (log(tan(lat) + sec(lat)) / pi)) / 2.0
    return x, y


def bounds_to_tiles(corner0, corner1):
    """Generates a 2d array of x, y pairs based on lat lon corners of a rectangle"""
    xy0 = lat_lon_to_tile(*corner0)
    xy1 = lat_lon_to_tile(*corner1)
    if xy1 < xy0:
        xy1, xy0 = xy0, xy1  # Sort

    # Expand bounding box and convert to int
    low_x, low_y = int(xy0[0]), int(xy0[1])
    hi_x, hi_y = int(xy1[0]), int(xy1[1])

    return np.mgrid[low_x:hi_x + 1, low_y:hi_y + 1].swapaxes(0, 2)


def save_tiles(tiles):
    """Download and save to disk new tiles if they don't already exist"""
    for tile in tiles.reshape(-1, tiles.shape[-1]):
        x, y = tile
        query = url.format(x, y)
        filename = fname_template.format(x, y)
        if os.path.exists(filename):
            continue
        urllib.request.urlretrieve(query, filename)


# https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays
def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return arr.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)


def rgb_to_meters(arr):
    """Convert an RGB array (image) to a 2d array of elevation values"""
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return (r * 256 + g + b / 256) - 32768


def join_tiles(tiles):
    """Generate a large 2d array of elevation measurements from a 2d array of tile locations"""

    def generate():
        """Helper to generate a list of single-tile elevation maps"""
        nrows, ncols, _ = tiles.shape
        # Iterate row by row
        for row in range(nrows):
            for col in range(ncols):
                tile = tiles[row, col, :]
                filename = fname_template.format(*tile)
                im = imageio.imread(filename)
                arr = np.array(im)
                yield rgb_to_meters(arr)

    arr = np.array(list(generate()))
    nh, nw, _ = tiles.shape  # Number of tiles in height and width
    h, w = arr[0].shape  # Size of an individual tile
    return unblockshaped(arr, h * nh, w * nw)


def get_full_map(corner0, corner1, trim=True):
    tiles = bounds_to_tiles(corner0, corner1)
    save_tiles(tiles)
    arr = join_tiles(tiles)

    if trim:
        # Trim the rectangle to only those measurements actually within the lat lon rectangle
        nw = lat_lon_to_tile(*corner0)
        se = lat_lon_to_tile(*corner1)
        if se < nw:  # Sort
            nw, se = se, nw

        nw_x, nw_y = nw
        se_x, se_y = se

        _, x_offset_low = int_and_dec(nw_x)
        _, y_offset_low = int_and_dec(nw_y)
        _, x_offset_hi = int_and_dec(se_x)
        _, y_offset_hi = int_and_dec(se_y)

        x_offset_low = int(x_offset_low * TILE_SIZE)
        y_offset_low = int(y_offset_low * TILE_SIZE)
        x_offset_hi = TILE_SIZE - int(x_offset_hi * TILE_SIZE)
        y_offset_hi = TILE_SIZE - int(y_offset_hi * TILE_SIZE)

        return arr[y_offset_low:-y_offset_hi, x_offset_low:-y_offset_hi]

    return arr


def normal(xyz0, xyz1, xyz2):
    a = xyz1 - xyz0
    b = xyz2 - xyz0
    return np.cross(a, b)


def write_triangle(fh, xyz0, xyz1, xyz2):
    norm = normal(xyz0, xyz1, xyz2)
    fh.write(struct.pack('<3f', *norm))
    fh.write(struct.pack('<3f', *xyz0))
    fh.write(struct.pack('<3f', *xyz1))
    fh.write(struct.pack('<3f', *xyz2))
    fh.write(struct.pack('<H', 0))


def surf_to_stl(arr, res, filename):
    with open(filename, 'wb') as f:
        # Dump 80 bytes of 0 as empty header
        f.write(struct.pack('80B', *[0 for _ in range(80)]))

        # Number of triangles
        nrows, ncols = arr.shape
        # Two triangles per square
        num_triangles = 2 * (nrows - 1) * (ncols - 1)
        # Same number of triangles on the bottom (could just be 2?)
        # num_triangles *= 2
        # Two triangles per pixel on the perimeter
        # num_triangles += 4 * nrows + 4 * ncols
        print(f'{num_triangles} triangles')
        f.write(struct.pack('<L', num_triangles))

        for row in range(nrows - 1):
            for col in range(ncols - 1):
                # First triangle _
                #               |/

                x00 = float(col) * res
                y00 = float(row) * res
                z00 = arr[row, col]
                xyz00 = np.array([x00, y00, z00])

                x01 = x00 + res
                y01 = y00
                z01 = arr[row, col + 1]
                xyz01 = np.array([x01, y01, z01])

                x10 = x00
                y10 = y00 + res
                z10 = arr[row + 1, col]
                xyz10 = np.array([x10, y10, z10])

                write_triangle(f, xyz00, xyz01, xyz10)

                # Second triangle /_|
                x11 = x00 + res
                y11 = y00 + res
                z11 = arr[row + 1, col + 1]
                xyz11 = np.array([x11, y11, z11])

                write_triangle(f, xyz11, xyz10, xyz01)


if __name__ == '__main__':
    nw = (46.931359, -121.898533)
    se = (46.761857, -121.624521)
    arr = get_full_map(nw, se, trim=False)

    print(arr.shape)

    res = resolution(ZOOM, (nw[0] + se[0]) / 2)

    scale = 2
    arr = arr[::scale, ::scale]
    res *= scale

    surf_to_stl(arr, res, 'rainier.stl')

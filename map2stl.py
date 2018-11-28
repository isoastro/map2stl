import numpy as np
import matplotlib.pyplot as plt
from math import radians, log, tan, cos, pi, floor, ceil
import urllib.request
import os
import imageio

## TODO: Generate STL file from 2d elevation data
## TODO: Also generate x, y data to account for mercator
    # This probably doesn't matter a ton, but it could be fun
## TODO: Make easier to use

# Because this doesn't exist otherwise...
def sec(x):
    return 1 / cos(x)

ZOOM = 13
url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{{:d}}/{{:d}}.png'
output_dir = 'tiles'
fname_template = os.path.join(output_dir, f'{ZOOM}_{{:d}}_{{:d}}.png')

def lat_lon_to_tile(latlon):
    '''Converts a given latitude and longitude (degrees) to a tile x, y'''
    lat, lon = latlon
    n = 2 ** ZOOM
    x = n * ((lon + 180.0) / 360.0)
    lat = radians(lat)
    y = n * (1 - (log(tan(lat) + sec(lat)) / pi)) / 2.0
    return x, y

def bounds_to_tiles(corner0, corner1):
    '''Generates a 2d array of x, y pairs based on lat lon corners of a rectangle'''
    xy0 = lat_lon_to_tile(corner0)
    xy1 = lat_lon_to_tile(corner1)
    if xy1 < xy0:
        xy1, xy0 = xy0, xy1 # Sort

    # Expand bounding box and convert to int
    xy0 = (int(floor(xy0[0])), int(floor(xy0[1])))
    xy1 = (int(ceil(xy1[0])), int(ceil(xy1[1])))

    return np.mgrid[xy0[0]:xy1[0]+1,xy0[1]:xy1[1]+1].swapaxes(0, 2)

def save_tiles(tiles):
    '''Download and save to disk new tiles if they don't already exist'''
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
    return arr.reshape(h//nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)

def rgb_to_meters(arr):
    '''Convert an RGB array (image) to a 2d array of elevation values'''
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    return (r * 256 + g + b / 256) - 32768

def big_array(tiles):
    '''Generate a large 2d array of elevation measurements from a 2d array of tile locations'''
    def generate():
        '''Helper to generate a list of single-tile elevation maps'''
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
    nh, nw, _ = tiles.shape # Number of tiles in height and width
    h, w = arr[0].shape # Size of an individual tile
    return unblockshaped(arr, h * nh, w * nw)

if __name__ == '__main__':
    nw = (46.986202, -121.947064)
    se = (46.754367, -121.574259)
    tiles = bounds_to_tiles(nw, se)
    save_tiles(tiles)
    
    arr = big_array(tiles)
    plt.imshow(arr, cmap='terrain')
    plt.show()

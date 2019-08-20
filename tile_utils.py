import numpy as np
import os.path
import urllib.request
import imageio
import matplotlib.pyplot as plt

TILE_SIZE = 256

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
    lat, lon = 46.852248, -121.757702  # Rainier
    zoom = 11

    lat = np.linspace(lat - 0.5, lat + 0.5, 9)
    lon = np.linspace(lon - 0.5, lon + 0.5, 5)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    lat_derived, lon_derived = tile_to_lat_lon(x, y, zoom)

    x_int = np.unique(x.astype(np.int))
    y_int = np.unique(y.astype(np.int))

    grid = build_grid(lat, lon, zoom)
    print(grid[1:10, 255])
    print(grid[1:10, 256])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(grid)
    # fig.show()
    plt.show()

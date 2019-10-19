import os.path
import urllib.request


def tile_to_url(x, y, zoom):
    return f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom:d}/{x:d}/{y:d}.png'


def tile_to_filename(x, y, zoom, output_dir='tiles'):
    return os.path.join(output_dir, f'{zoom:d}_{x:d}_{y:d}.png')


def save_tile(x, y, zoom):
    url = tile_to_url(x, y, zoom)
    filename = tile_to_filename(x, y, zoom)
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f'Saved {url} to {filename}', flush=True)
    return filename

import numpy as np
import struct
import tile
import argparse


def normal(xyz0, xyz1, xyz2):
    a = xyz1 - xyz0
    b = xyz2 - xyz0
    return np.cross(a, b)


def write_triangle(fh ,xyz0, xyz1, xyz2):
    norm = normal(xyz0, xyz1, xyz2)
    fh.write(struct.pack('<3f', *norm))
    fh.write(struct.pack('<3f', *xyz0))
    fh.write(struct.pack('<3f', *xyz1))
    fh.write(struct.pack('<3f', *xyz2))
    fh.write(struct.pack('<H', 0)) # Weird 16-bit number that doesn't do anything(?)


def xyz_to_stl(xyz, filename):
    with open(filename, 'wb') as f:
        # Dump 80 bytes of 0 as empty header
        f.write(struct.pack('80B', *[0 for _ in range(80)]))

        # Number of triangles
        _, nrows, ncols = xyz.shape
        # Two triangles per square
        num_triangles = 2 * (nrows - 1) * (ncols - 1)
        print(f'Writing {num_triangles} to {filename}')
        f.write(struct.pack('<L', num_triangles))

        for i in range(nrows - 1):
            for j in range(ncols - 1):
                # First triangle _
                #               |/
                xyz00 = xyz[:, i, j]
                xyz01 = xyz[:, i, j+1]
                xyz10 = xyz[:, i+1, j]
                write_triangle(f, xyz00, xyz01, xyz10)

                # Second triangle /_|
                xyz11 = xyz[:, i+1, j+1]
                write_triangle(f, xyz11, xyz10, xyz01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corner1', nargs=2, type=float)
    parser.add_argument('corner2', nargs=2, type=float)
    parser.add_argument('zoom', type=int)
    parser.add_argument('output')

    args = parser.parse_args()

    # Create map
    tmap = tile.TileMap(args.corner1, args.corner2, args.zoom)
    tmap.scale(1 / tmap.resolution(args.zoom))

    xyz_to_stl(tmap.xyz, args.output)


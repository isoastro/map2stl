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


def get_header(args):
    lat1, lon1 = args.corner1
    lat2, lon2 = args.corner2
    header = f'({lat1}, {lon1}) - ({lat2}, {lon2}) @ zoom={args.zoom}'
    header_len = len(header)
    if header_len > 80:
        raise ValueError('Tried to create header larger than 80 bytes')
    return header.encode() + struct.pack(f'{80 - header_len}B', *[0 for _ in range(80 - header_len)])


def xyz_to_stl(xyz, args):
    # TODO: Make actually solid by dropping down to sea level or configurable depth and closing out
    _, nrows, ncols = xyz.shape
    # Two triangles per square
    num_triangles = 2 * (nrows - 1) * (ncols - 1)
    print(f'Writing {num_triangles} triangles to {args.output}', flush=True)

    with open(args.output, 'wb') as f:
        # Dump 80 bytes of 0 as empty header
        f.write(get_header(args))

        # Dump number of triangles
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
    # TODO: Add different options other than just specifying corners. Like center and radius, center and square lengths
    # TODO: Add argument for resolution, and pick zoom level from that. Decimate array if meters/pixel is between levels

    # TODO: There's a bug when run with the following arguments:
    # 42.989719 -122.184255 42.880737 -122.016995 12 crater_lake.stl
    # The correct tiles are downloaded, but the STL file is missing part of the lake. Probably something wrong when
    # trimming the data

    args = parser.parse_args()

    # Create map
    tmap = tile.TileMap(args.corner1, args.corner2, args.zoom)

    # Dump to STL
    xyz_to_stl(tmap.xyz, args)


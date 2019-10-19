import numpy as np
import struct
import tile
import argparse
import os.path


def progress_bar(iteration, total, length=50):
    percent = (iteration * 1000) // total
    if percent == progress_bar.last_percent:
        return
    progress_bar.last_percent = percent
    filled_length = int(length * iteration // total)
    bar = 'X' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {iteration / total:.1%} ({iteration:{len(str(total))}}/{total})', end='\r', flush=True)
    if iteration >= total:
        print()
progress_bar.last_percent = None


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
    header = f'({lat1},{lon1})-({lat2}, {lon2}) @ zoom={args.zoom}; raw data ©Mapzen'
    header = header.encode()
    header_len = len(header)
    if header_len > 80:
        raise ValueError(f'Tried to create {header_len} byte long header (max 80)')
    return header + struct.pack(f'{80 - header_len}B', *[0 for _ in range(80 - header_len)])


def filesize(num_triangles):
    header = 80 + 4
    per_tri = (4 * 3 * 4) + 2
    return header + (num_triangles * per_tri)


def xyz_to_stl(xyz, args):
    # TODO: Make actually solid by dropping down to sea level or configurable depth and closing out
    _, nrows, ncols = xyz.shape

    num_triangles = 2 * (nrows - 1) * (ncols - 1)  # Two triangles per square
    num_triangles *= 2  # Duplicated on bottom
    num_triangles += (nrows - 1) * 2 * 2  # Edges down to floor
    num_triangles += (ncols - 1) * 2 * 2

    min_height = xyz[2, :, :].min()
    floor = args.floor
    if floor < 0:
        floor = min_height + floor
    bottom = np.stack((*xyz[0:2, :, :], floor * np.ones((nrows, ncols))))

    MB = filesize(num_triangles) / 1024 / 1024
    print(f'Writing {num_triangles} triangles to {args.output} ({MB:.2f} MiB)', flush=True)

    directory = os.path.dirname(args.output)
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_triangles_written = 0
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

                # First triangle on bottom
                b00 = bottom[:, i, j]
                b01 = bottom[:, i, j+1]
                b10 = bottom[:, i+1, j]
                write_triangle(f, b00, b01, b10)

                # Second triangle on bottom
                b11 = bottom[:, i+1, j+1]
                write_triangle(f, b11, b10, b01)

                num_triangles_written += 4
                progress_bar(num_triangles_written, num_triangles)

        # Edges
        for i in range(nrows - 1):
            A = xyz[:, i, 0]
            B = bottom[:, i, 0]
            C = xyz[:, i+1, 0]
            D = bottom[:, i+1, 0]
            write_triangle(f, A, B, D)
            write_triangle(f, A, D, C)

            A = xyz[:, i, -1]
            B = bottom[:, i, -1]
            C = xyz[:, i+1, -1]
            D = bottom[:, i+1, -1]
            write_triangle(f, A, B, D)
            write_triangle(f, A, D, C)

            num_triangles_written += 4
            progress_bar(num_triangles_written, num_triangles)

        for i in range(ncols - 1):
            A = xyz[:, 0, i]
            B = bottom[:, 0, i]
            C = xyz[:, 0, i+1]
            D = bottom[:, 0, i+1]
            write_triangle(f, A, B, D)
            write_triangle(f, A, D, C)

            A = xyz[:, -1, i]
            B = bottom[:, -1, i]
            C = xyz[:, -1, i+1]
            D = bottom[:, -1, i+1]
            write_triangle(f, A, B, D)
            write_triangle(f, A, D, C)

            num_triangles_written += 4
            progress_bar(num_triangles_written, num_triangles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corner1', nargs=2, type=float)
    parser.add_argument('corner2', nargs=2, type=float)
    parser.add_argument('zoom', type=int)
    parser.add_argument('output')
    parser.add_argument('--floor', nargs='?', type=float, default=0.0)
    # TODO: Add different options other than just specifying corners. Like center and radius, center and square lengths
    # TODO: Add argument for resolution, and pick zoom level from that. Decimate array if meters/pixel is between levels

    args = parser.parse_args()

    # Create map
    tmap = tile.TileMap(args.corner1, args.corner2, args.zoom)

    # Dump to STL
    xyz_to_stl(tmap.xyz, args)


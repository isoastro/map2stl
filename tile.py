import numpy as np
import imageio
import api

TILE_SIZE = 256
WGS84_RADIUS = 6378137  # Meters
WGS84_CIRCUMFERENCE = 2 * np.pi * WGS84_RADIUS
TILE_RESOLUTION = WGS84_CIRCUMFERENCE / TILE_SIZE
ORIGIN_METERS = WGS84_CIRCUMFERENCE / 2


class TileMap:
    MERCATOR_X = 0
    MERCATOR_Y = 1
    ELEVATION = 2
    DATA_DEPTH = 3

    def __init__(self, corner1, corner2, zoom):
        self.zoom = zoom
        self._tiles = self.corners_to_tiles(corner1, corner2, zoom)
        num_x_pixels = len(self._tiles) * TILE_SIZE
        num_y_pixels = len(self._tiles[0]) * TILE_SIZE
        self._data = np.zeros((self.DATA_DEPTH, num_x_pixels, num_y_pixels))

        # Get all the data
        self.download_tiles()

        # Recalculate pixel space
        # TODO: Is this correct? Where is the actual origin? Top left of each tile? What pixel/meter does that correspond to?
        ox, oy, _ = self._tiles[0][0] # Origin
        ox, oy = self.tile_coords_to_pixel(ox, oy)
        pixels = np.mgrid[ox:ox + num_x_pixels, oy:oy + num_y_pixels]

        # Back out to meters
        self._data[self.MERCATOR_X:self.MERCATOR_Y+1, :, :] = self.pixel_to_mercator(*pixels, self.zoom)

        # Trim data and make x, y start at 0
        self.trim(corner1, corner2)
        self.zeroize_xy()

    def trim(self, corner1, corner2):
        lat_bounds = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
        lon_bounds = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
        lat_within_bounds = np.logical_and(
            self.latlon[0] >= lat_bounds[0],
            self.latlon[0] <= lat_bounds[1],
        )
        lon_within_bounds = np.logical_and(
            self.latlon[1] >= lon_bounds[0],
            self.latlon[1] <= lon_bounds[1],
        )

        mask = np.ix_(np.arange(self.DATA_DEPTH), lat_within_bounds, lon_within_bounds)
        self._data = self._data[mask]

    def zeroize_xy(self):
        self._data[self.MERCATOR_X, :, :] -= self.mercator[0].min()
        self._data[self.MERCATOR_Y, :, :] -= self.mercator[1].min()

    def scale(self, scale_factor):
        self._data *= scale_factor

    @property
    def elevation(self):
        return self._data[self.ELEVATION, :, :]

    @property
    def mercator(self):
        return self._data[self.MERCATOR_X:self.MERCATOR_Y+1, :, :]

    @property
    def xyz(self):
        return self._data[self.MERCATOR_X:self.ELEVATION+1, :, :]

    @property
    def latlon(self):
        mx = self.mercator[0, :, 0]
        my = self.mercator[1, 0, :]
        return np.array(self.mercator_to_latlon(mx, my))

    # TODO: Make this take an arbitrary polygon and generate all tiles the polygon overlaps
    @classmethod
    def corners_to_tiles(cls, corner1, corner2, zoom):
        # Convert corners to tile coordinates
        mercator1 = cls.latlon_to_mercator(*corner1)
        pixel1 = cls.mercator_to_pixel(*mercator1, zoom)
        tile_coords1 = cls.pixel_to_tile_coords(*pixel1)
        tile_coords1 = [int(_) for _ in tile_coords1]

        mercator2 = cls.latlon_to_mercator(*corner2)
        pixel2 = cls.mercator_to_pixel(*mercator2, zoom)
        tile_coords2 = cls.pixel_to_tile_coords(*pixel2)
        tile_coords2 = [int(_) for _ in tile_coords2]

        lo_x = min(tile_coords1[0], tile_coords2[0])
        hi_x = max(tile_coords1[0], tile_coords2[0])
        lo_y = min(tile_coords1[1], tile_coords2[1])
        hi_y = max(tile_coords1[1], tile_coords2[1])

        return [[(x, y, zoom) for x in range(lo_x, hi_x + 1)] for y in range(lo_y, hi_y + 1)]

    @staticmethod
    def resolution(zoom):
        return TILE_RESOLUTION / (2 ** zoom)

    @staticmethod
    def latlon_to_mercator(lat, lon):
        mx = WGS84_RADIUS * np.radians(lon)
        my = WGS84_RADIUS * -np.log(np.tan((np.pi / 4) + (np.radians(lat) / 2)))
        return mx, my

    @staticmethod
    def mercator_to_latlon(mx, my):
        lon = np.degrees(mx / WGS84_RADIUS)
        lat = -np.degrees(2 * np.arctan(np.exp(my/WGS84_RADIUS)) - (np.pi / 2))
        return lat, lon

    @classmethod
    def mercator_to_pixel(cls, mx, my, zoom):
        r = cls.resolution(zoom)
        px = (mx + ORIGIN_METERS) / r
        py = (my + ORIGIN_METERS) / r
        return px, py

    @classmethod
    def pixel_to_mercator(cls, px, py, zoom):
        r = cls.resolution(zoom)
        mx = px * r - ORIGIN_METERS
        my = py * r - ORIGIN_METERS
        return mx, my

    @staticmethod
    def pixel_to_tile_coords(px, py):
        tx = (px / TILE_SIZE)
        ty = (py / TILE_SIZE)
        return tx, ty

    @staticmethod
    def tile_coords_to_pixel(tx, ty):
        px = (tx) * TILE_SIZE
        py = (ty) * TILE_SIZE
        return px, py

    @staticmethod
    def rgb_to_meters(rgb):
        return (rgb[:, :, 0] * 256 + rgb[:, :, 1] + rgb[:, :, 2] / 256) - 32768

    def download_tiles(self):
        for i, row in enumerate(self._tiles):
            for j, tile in enumerate(row):
                filename = api.save_tile(*tile)
                arr = np.array(imageio.imread(filename))
                height = self.rgb_to_meters(arr)
                ii = np.s_[i * TILE_SIZE:i * TILE_SIZE + TILE_SIZE]
                jj = np.s_[j * TILE_SIZE:j * TILE_SIZE + TILE_SIZE]
                self._data[self.ELEVATION, ii, jj] = height


if __name__ == '__main__':
    se = (46.925229, -121.829145) # Rainier
    nw = (46.762427, -121.632913) # Rainier
    # se = (47.497631, -122.185169) # Seattle
    # nw = (47.739248, -122.448340) # Seattle
    z = 8

    t = TileMap(se, nw, z)
    print('min lat, lon', t.latlon[0].min(), t.latlon[1].min())
    print('max lat, lon', t.latlon[0].max(), t.latlon[1].max())

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*t.mercator, t.elevation)
    plt.show()
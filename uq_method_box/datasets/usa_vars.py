"""USA Vars Dataset for OOD Tasks."""

from typing import Any, Callable, Dict, Optional, Sequence

import geopandas
import geoplot as gplt
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset, USAVars

# TODO
# 97871 images
# Come up with new loading strategy for the corresponding split
# 1. Come up with a useful OOD split, EAST/WEST or big Checkerboard split?
#    - or define range values on the target
# 2. Visualize the split with lat lon locations
# 3. Labels contain an id to the imagery so can also work backwards from
#   label to img to decide upon a split / probably cheaper than with Raster

# QUESTIONS:
# 1. How to constrain problem for treecover percentage as a percentage prediction with UQ
# 2. Should come up with a fixed dataset version for reproducibility

ds = USAVars(
    root="/home/nils/projects/uq-regression-box/experiments/data/usa_vars/",
    labels=["treecover"],
)

fig, axs = plt.subplots(ncols=3)
for idx, i in enumerate([57, 7862, 11728]):
    sample = ds[i]
    plt.sca(axs[idx])
    ds.plot(sample, axs=axs[idx])

import pdb

pdb.set_trace()


class USAVarsRaster(RasterDataset):
    """Raster Dataset to utitlize the geo information."""

    def __init__(
        self,
        root: str = "data",
        crs=None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)

        filename_glob = "tile_*.tif"
        filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
        is_image = True
        separate_files = False


# ds = USAVarsRaster(
#     root="/home/nils/projects/uq-regression-box/experiments/data/usa_vars/uar"
# )


class USAVarsOOD(USAVars):
    """USA Vars Dataset adapted for OOD."""

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        labels: Sequence[str] = ["treecover"],
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        super().__init__(root, split, labels, None, download, checksum)

        # how to get the ood splits we want

        # adapt self.files and self.label_dfs
        treecover_df = self.label_dfs["treecover"]

        treecover_gdf = geopandas.GeoDataFrame(
            treecover_df,
            geometry=geopandas.points_from_xy(treecover_df.lon, treecover_df.lat),
        )

        n = 20000
        ax = gplt.pointplot(
            treecover_gdf.sample(n), hue="treecover", legend=True, figsize=(16, 12)
        )

        ax.set_title(
            f"Spatial Distribution of Treecover for {n} randomly sampled points."
        )

        import pdb

        pdb.set_trace()

        print(0)


ds = USAVarsOOD(root="/home/nils/projects/uq-regression-box/experiments/data/usa_vars/")

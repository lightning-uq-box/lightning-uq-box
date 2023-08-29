target = self.dataframe["percent(t)"]
import geopandas
import geoplot as gplt
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2)

gdf = geopandas.GeoDataFrame(
    target, geometry=geopandas.points_from_xy(self.dataframe.lat, self.dataframe.lon)
)
contiguous_usa = geopandas.read_file(gplt.datasets.get_path("contiguous_usa"))
gplt.polyplot(contiguous_usa, ax=ax[0])
gplt.pointplot(gdf, hue="percent(t)", legend=True, ax=ax[0])

ax[0].set_title(f"Spatial Distribution of Live Fuel Moisture for the full dataset.")

ax[1].violinplot(self.dataframe["percent(t)"].values)
ax[1].set_title("Distribution of Moisture Percentage Full dataset.")
ax[1].set_ylabel("Live Fuel Moisture Percentage")
import pdb

pdb.set_trace()
print(0)

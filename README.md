# xoverturning
MOM6 ocean overturning in xarray

## QuickStart guide:

* Import a dataset containing the grid and transports:

```python
import xarray as xr
import matplotlib.pyplot as plt

ppdir = '/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20201120/CM4_piControl_c96_OM4p25_half_kdadd/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual_z'
ds = xr.open_mfdataset([f"{ppdir}/ocean_annual_z.static.nc",
                       f"{ppdir}/ts/annual/10yr/ocean_annual_z.0021-0030.umo.nc",
                       f"{ppdir}/ts/annual/10yr/ocean_annual_z.0021-0030.vmo.nc",
                       f"{ppdir}/ts/annual/10yr/ocean_annual_z.0021-0030.uhml.nc",
                       f"{ppdir}/ts/annual/10yr/ocean_annual_z.0021-0030.vhml.nc"])

hgrid = xr.open_dataset('/archive/gold/datasets/OM4_025/mosaic.v20170622.unpacked/ocean_hgrid.nc')
ds['angle_dx'] = xr.DataArray(hgrid['angle_dx'].values[1::2,1::2], dims=('yh','xh'))
```

 * Compute your favorite MOC:

```python
from xoverturning import calcmoc

# global MOC
moc = calcmoc(ds)
# atlantic
moc = calcmoc(ds, basin='atl-arc')
# remove thickness flux to mixed layer
moc = calcmoc(ds, basin='atl-arc', remove_hml=True)
# rotate velocities to True North
moc = calcmoc(ds, basin='atl-arc', rotate=True)
# in density coordinates
moc = calcmoc(ds, vertdim='rho2_l')
```

## Computing derived quantities:

* Max MOC:

```python
maxmoc = moc.max(dim=['yh', 'z_l'])
```

* Min MOC:

```python
maxmoc = moc.min(dim=['yh', 'z_l'])
```

* Max MOC at 26.5N:

```python
maxmoc_265 = moc.sel(yh=slice(26.4,26.6)).max(dim=['z_l'])
```


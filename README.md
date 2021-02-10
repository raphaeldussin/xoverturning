# xoverturning
ocean overturning in xarray

* QuickStart guide:

Import a dataset containing the grid and transports:

```python
import xarray as xr
import matplotlib.pyplot as plt
ds = xr.open_mfdataset(['/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20201120/CM4_piControl_c96_OM4p2
5_half_kdadd/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual_z/ocean_annual_z.static.nc', '/archive/Raph
ael.Dussin/FMS2019.01.03_devgfdl_20201120/CM4_piControl_c96_OM4p25_half_kdadd/gfdl.ncrc4-intel18-prod-o
penmp/pp/ocean_annual_z/ts/annual/10yr/ocean_annual_z.0021-0030.umo.nc', '/archive/Raphael.Dussin/FMS20
19.01.03_devgfdl_20201120/CM4_piControl_c96_OM4p25_half_kdadd/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_a
nnual_z/ts/annual/10yr/ocean_annual_z.0021-0030.vmo.nc', '/archive/Raphael.Dussin/FMS2019.01.03_devgfdl
_20201120/CM4_piControl_c96_OM4p25_half_kdadd/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual_z/ts/annua
l/10yr/ocean_annual_z.0021-0030.uhml.nc', '/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20201120/CM4_p
iControl_c96_OM4p25_half_kdadd/gfdl.ncrc4-intel18-prod-openmp/pp/ocean_annual_z/ts/annual/10yr/ocean_an
nual_z.0021-0030.vhml.nc'])
hgrid = xr.open_dataset('/archive/gold/datasets/OM4_025/mosaic.v20170622.unpacked/ocean_hgrid.nc')
ds['angle_dx'] = xr.DataArray(hgrid['angle_dx'].values[1::2,1::2], dims=('yh','xh'))
```

Compute your favorite MOC:

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

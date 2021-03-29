import numpy as np
import pytest
import xarray as xr
import xgcm

from xoverturning.moc import define_names

ds_sym = xr.Dataset()
ds_sym["xh"] = xr.DataArray(np.arange(10), dims=("xh"))
ds_sym["xq"] = xr.DataArray(np.arange(11), dims=("xq"))
ds_sym["yh"] = xr.DataArray(np.arange(20), dims=("yh"))
ds_sym["yq"] = xr.DataArray(np.arange(21), dims=("yq"))
ds_sym["z_l"] = xr.DataArray(np.arange(5), dims=("z_l"))
ds_sym["z_i"] = xr.DataArray(np.arange(6), dims=("z_i"))
ds_sym["umo"] = xr.DataArray(
    np.random.rand(1, 5, 20, 11), dims=("time", "z_l", "yh", "xq")
)
ds_sym["uhml"] = xr.DataArray(
    np.random.rand(1, 5, 20, 11), dims=("time", "z_l", "yh", "xq")
)
ds_sym["vmo"] = xr.DataArray(
    np.random.rand(1, 5, 21, 10), dims=("time", "z_l", "yq", "xh")
)
ds_sym["vhml"] = xr.DataArray(
    np.random.rand(1, 5, 21, 10), dims=("time", "z_l", "yq", "xh")
)

ds_asym = xr.Dataset()
ds_asym["xh"] = xr.DataArray(np.arange(10), dims=("xh"))
ds_asym["xq"] = xr.DataArray(np.arange(10), dims=("xq"))
ds_asym["yh"] = xr.DataArray(np.arange(20), dims=("yh"))
ds_asym["yq"] = xr.DataArray(np.arange(20), dims=("yq"))
ds_asym["z_l"] = xr.DataArray(np.arange(5), dims=("z_l"))
ds_asym["z_i"] = xr.DataArray(np.arange(6), dims=("z_i"))
ds_asym["umo"] = xr.DataArray(
    np.random.rand(1, 5, 20, 10), dims=("time", "z_l", "yh", "xq")
)
ds_asym["uhml"] = xr.DataArray(
    np.random.rand(1, 5, 20, 10), dims=("time", "z_l", "yh", "xq")
)
ds_asym["vmo"] = xr.DataArray(
    np.random.rand(1, 5, 20, 10), dims=("time", "z_l", "yq", "xh")
)
ds_asym["vhml"] = xr.DataArray(
    np.random.rand(1, 5, 20, 10), dims=("time", "z_l", "yq", "xh")
)

ds_wrong = xr.Dataset()
ds_wrong["xh"] = xr.DataArray(np.arange(10), dims=("xh"))
ds_wrong["xq"] = xr.DataArray(np.arange(10), dims=("xq"))
ds_wrong["yh"] = xr.DataArray(np.arange(20), dims=("yh"))
ds_wrong["yq"] = xr.DataArray(np.arange(21), dims=("yq"))

ds_angle1 = xr.Dataset()
ds_angle1["angle_dx"] = xr.DataArray(
    (90 * np.pi / 180) * np.ones([20, 10]), dims=("yh", "xh")
)

ds_angle2 = xr.Dataset()
ds_angle2["cos_rot"] = np.cos(ds_angle1["angle_dx"])
ds_angle2["sin_rot"] = np.sin(ds_angle1["angle_dx"])


woa_opendap = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/temperature/decav/1.00/woa18_decav_t00_01.nc"
try:
    ds_1x1deg = xr.open_dataset(woa_opendap, decode_times=False, engine="pydap")
except:
    ds_1x1deg = None

if ds_1x1deg is not None:
    ds_1x1deg = ds_1x1deg.rename({"lon": "xh", "lat": "yh", "depth": "z_l"})
    ds_1x1deg["z_i"] = xr.DataArray(
        np.concatenate([[0], ds_1x1deg["depth_bnds"].isel(nbounds=1)]), dims=("z_i")
    )
    ds_1x1deg["xq"] = ds_1x1deg["xh"] + 0.5
    ds_1x1deg["yq"] = ds_1x1deg["yh"] + 0.5
    lon, lat = np.meshgrid(ds_1x1deg["xh"], ds_1x1deg["yh"])
    ds_1x1deg["lon"] = xr.DataArray(lon, dims=("yh", "xh"))
    ds_1x1deg["lat"] = xr.DataArray(lat, dims=("yh", "xh"))
    ds_1x1deg = ds_1x1deg.set_coords(["xh", "yh", "xq", "yq"])
    ds_1x1deg["wet"] = xr.where(
        ds_1x1deg["t_an"].isel(z_l=0, time=0).fillna(9999) == 9999, 0, 1
    )
    ds_1x1deg["deptho"] = 4000.0 * ds_1x1deg["wet"]
    ds_1x1deg["umo"] = xr.DataArray(
        np.random.rand(1, 102, 180, 360), dims=("time", "z_l", "yh", "xq")
    )
    ds_1x1deg["uhml"] = xr.DataArray(
        np.random.rand(1, 102, 180, 360), dims=("time", "z_l", "yh", "xq")
    )
    ds_1x1deg["vmo"] = xr.DataArray(
        np.random.rand(1, 102, 180, 360), dims=("time", "z_l", "yq", "xh")
    )
    ds_1x1deg["vhml"] = xr.DataArray(
        np.random.rand(1, 102, 180, 360), dims=("time", "z_l", "yq", "xh")
    )
    # ds_1x1deg.to_netcdf("testwoa.nc")


def test_is_symetric():
    from xoverturning.compfunc import is_symetric

    names = define_names(model="mom6", vertical="z")
    assert is_symetric(ds_sym, names)
    assert not is_symetric(ds_asym, names)
    with pytest.raises(ValueError):
        is_symetric(ds_wrong, names)


@pytest.mark.parametrize("DS", [ds_sym, ds_asym])
def test_define_grid(DS):
    from xoverturning.compfunc import define_grid

    names = define_names(model="mom6", vertical="z")
    grid = define_grid(DS, names)
    assert isinstance(grid, xgcm.Grid)


@pytest.mark.parametrize("DS", [ds_sym, ds_asym])
def test_substract_hml(DS):
    from xoverturning.compfunc import substract_hml

    u, v = substract_hml(DS)
    assert isinstance(u, xr.core.dataarray.DataArray)
    assert isinstance(v, xr.core.dataarray.DataArray)

    with pytest.raises(IOError):
        u, v = substract_hml(DS.drop_vars(["uhml", "vhml"]))


@pytest.mark.parametrize("DS", [ds_sym, ds_asym])
@pytest.mark.parametrize("DSANGLE", [ds_angle1, ds_angle2])
def test_rotate_velocities_to_geo(DS, DSANGLE):
    from xoverturning.compfunc import rotate_velocities_to_geo

    names = define_names(model="mom6", vertical="z")
    with pytest.raises(ValueError):
        u_EW, v_EW = rotate_velocities_to_geo(DS, DS["umo"], DS["vmo"], names)

    ds = xr.merge([DS, DSANGLE])

    u_EW, v_EW = rotate_velocities_to_geo(ds, ds["umo"], ds["vmo"], names)
    assert isinstance(u_EW, xr.core.dataarray.DataArray)
    assert isinstance(v_EW, xr.core.dataarray.DataArray)
    assert "yh" in u_EW.dims
    assert "xh" in u_EW.dims
    assert "yh" in v_EW.dims
    assert "xh" in v_EW.dims


@pytest.mark.parametrize("DS", [ds_sym, ds_asym])
def test_interp_to_grid_center(DS):
    from xoverturning.compfunc import interp_to_grid_center

    names = define_names(model="mom6", vertical="z")
    u, v = interp_to_grid_center(DS, DS["umo"], DS["vmo"], names)
    assert isinstance(u, xr.core.dataarray.DataArray)
    assert isinstance(v, xr.core.dataarray.DataArray)
    assert "yh" in u.dims
    assert "xh" in u.dims
    assert "yh" in v.dims
    assert "xh" in v.dims


@pytest.mark.parametrize("BASIN", ["global", "atl-arc", "indopac"])
def test_select_basin(BASIN):
    from xoverturning.compfunc import select_basins

    if ds_1x1deg is None:
        return

    names = define_names(model="mom6", vertical="z")
    maskbasin, maskmoc = select_basins(
        ds_1x1deg, names, basin=BASIN, lon="lon", lat="lat"
    )
    assert isinstance(maskbasin, xr.core.dataarray.DataArray)
    assert isinstance(maskmoc, xr.core.dataarray.DataArray)

    maskbasin, maskmoc = select_basins(
        ds_1x1deg, names, basin=BASIN, lon="lon", lat="lat", vertical="rho2"
    )
    assert maskmoc is None


@pytest.mark.parametrize("DS", [ds_sym, ds_asym])
def test_compute_streamfunction(DS):
    from xoverturning.compfunc import compute_streamfunction

    names = define_names(model="mom6", vertical="z")
    psi = compute_streamfunction(DS, names, transport="vmo")

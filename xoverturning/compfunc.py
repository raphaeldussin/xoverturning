import numpy as np
import xarray as xr
from xgcm import Grid
import warnings
from cmip_basins import generate_basin_codes


def is_symetric(ds):
    """check if grid is symetric

    Args:
        ds (xarray.Dataset): dataset containing model's grid

    Returns:
        bool: True if grid is symetric
    """

    if (len(ds["xq"]) == len(ds["xh"])) and (len(ds["yq"]) == len(ds["yh"])):
        out = False
    elif (len(ds["xq"]) == len(ds["xh"]) + 1) and (len(ds["yq"]) == len(ds["yh"]) + 1):
        out = True
    else:
        raise ValueError("unsupported combination of coordinates")
    return out


def define_grid(ds):
    """build a xgcm.Grid object

    Args:
        ds (xarray.Dataset): dataset with model's grid

    Returns:
        xgcm.Grid: grid object
    """

    qcoord = "outer" if is_symetric(ds) else "right"

    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", qcoord: "xq"},
            "Y": {"center": "yh", qcoord: "yq"},
        },
        periodic=["X"],
    )
    return grid


def substract_hml(ds, umo="umo", vmo="vmo", uhml="uhml", vhml="vhml"):
    """substracting Thickness Flux to Restratify Mixed Layer
    from transports

    Args:
        ds (xarray.Dataset): dataset containing transports
        umo (str, optional): name of zonal transport
        vmo (str, optional): name of meridional transport
        uhml (str, optional): name of zonal Thickness Flux
        vhml (str, optional): name of meriodional Thickness Flux

    Returns:
        xarray.DataArray: corrected transports
    """

    if uhml in ds.variables:
        # substract from meridional transport
        ucorr = ds[umo] - ds[uhml]
    else:
        warnings.warn(f"{uhml} not found, not substracting")
        ucorr = ds[umo]

    if vhml in ds.variables:
        # substract from meridional transport
        vcorr = ds[vmo] - ds[vhml]
    else:
        warnings.warn(f"{vhml} not found, not substracting")
        vcorr = ds[vmo]

    return ucorr, vcorr


def rotate_velocities_to_geo(ds, da_u, da_v):
    """rotate a pair of velocity vectors to the geographical axes

    Args:
        ds (xarray.Dataset): dataset containing velocities to rotate
        da_u (xarray.DataAray): data for u-component of velocity in model coordinates
        da_v (xarray.DataArray): data for v-component of velocity in model coordinates

    Returns:
        xarray.DataArray: rotated velocities
    """

    if "cos_rot" in ds.variables and "sin_rot" in ds.variables:
        CS = ds["cos_rot"]
        SN = ds["sin_rot"]
    elif "angle_dx" in ds.variables:
        CS = np.cos(ds["angle_dx"])
        SN = np.sin(ds["angle_dx"])
    else:
        # I would like to have a way to retrieve angle from lon/lat arrays
        raise ValueError("angle or components must be included in dataset")

    # build the xgcm grid object
    grid = define_grid(ds)
    # interpolate to the cell centers
    u_ctr = grid.interp(da_u, "X")
    v_ctr = grid.interp(da_v, "Y")
    # rotation inverse from the model's grid angle
    u_EW = u_ctr * CS - v_ctr * SN
    v_EW = v_ctr * CS + u_ctr * SN

    return u_EW, v_EW


def interp_to_grid_center(ds, da_u, da_v):
    """interpolate velocities to cell centers

    Args:
        ds (xarray.Dataset): dataset containing velocities to rotate
        da_u (xarray.DataAray): data for u-component of velocity in model coordinates
        da_v (xarray.DataArray): data for v-component of velocity in model coordinates

    Returns:
        xarray.DataArray: interpolated velocities
    """
    # build the xgcm grid object
    grid = define_grid(ds)
    # interpolate to the cell centers
    u_ctr = grid.interp(da_u, "X", boundary="fill")
    v_ctr = grid.interp(da_v, "Y", boundary="fill")
    return u_ctr, v_ctr


def select_basins(
    ds,
    basin="global",
    lon="geolon",
    lat="geolat",
    mask="wet",
    bathy="deptho",
    verbose=True,
):
    """generate a mask for selected basin

    Args:
        ds (xarray.Dataset): dataset contaning model grid
        basin (str, optional): global/atl-arc/indopac. Defaults to "global".
        lon (str, optional): name of geographical lon in dataset. Defaults to "geolon".
        lat (str, optional): name of geographical lat in dataset. Defaults to "geolat".
        mask (str, optional): name of land/sea mask in dataset. Defaults to "wet".
        bathy (str, optional): name of bathymetry in dataset. Defaults to "deptho".
        verbose (bool, optional): Verbose output. Defaults to True.

    Returns:
        xarray.DataArray: mask for selected basin
    """

    # read or recalculate basin codes
    if "basin" in ds:
        basincodes = ds["basin"]
    else:
        if verbose:
            print("generating basin codes")
        basincodes = generate_basin_codes(ds, lon=lon, lat=lat, mask=mask)

    # expand land sea mask to remove other basins
    if basin == "global":
        maskbin = ds[mask]
    elif basin == "atl-arc":
        maskbin = ds[mask].where((basincodes == 2) | (basincodes == 4))
    elif basin == "indopac":
        maskbin = ds[mask].where((basincodes == 3) | (basincodes == 5))

    maskbasin = xr.where(maskbin == 1, True, False)

    # create a mask for the streamfunction
    if bathy in ds:
        if "yq" in maskbasin.dims:
            grid = define_grid(ds)
            bathy_coloc = grid.interp(ds[bathy], "Y", boundary="fill")
        elif "yh" in maskbasin.dims:
            bathy_coloc = ds[bathy]
        else:
            raise ValueError("Unsupported coord")
        bathy_basin = bathy_coloc.where(maskbasin).fillna(0.0)
        max_depth = bathy_basin.max(dim="xh")
        maskmoc = xr.where(ds["z_i"] > max_depth, 0, 1)
    else:
        maskmoc = None

    return maskbasin, maskmoc


def compute_streamfunction(
    ds,
    xdim="xh",
    layer="z_l",
    interface="z_i",
    rho0=1035.0,
    add_offset=False,
    offset=0.1,
):
    """compute the overturning streamfunction from meridional transport

    Args:
        ds (xarray.Dataset): meridional transport in kg.s-1
        xdim (str, optional): name of zonal dimension. Defaults to "xh".
        zdim (str, optional): name of the vertical dimension (e.g. z_l, rho2_l). Defaults to "z_l".
        rho0 (float, optional): average density of seawater. Defaults to 1035.0.
        add_offset (bool, optional): add a small number to clean 0 contours. Defaults to False.
        offset (float, optional): offset for contours, should be small. Defaults to 0.1.

    Returns:
        xarray.DataArray: Overturning streamfunction
    """

    # sum over the zonal direction
    zonalsum = ds["v"].sum(dim=xdim)
    # integrate from bottom
    integ_layers_from_bottom = zonalsum.cumsum(dim=layer) - zonalsum.sum(dim=layer)
    # the result of the integration over layers is evaluated at the interfaces
    # with psi = 0 as the bottom boundary condition for the integration
    bottom_condition = xr.zeros_like(integ_layers_from_bottom.isel({layer: -1}))
    psi_raw = xr.concat([integ_layers_from_bottom, bottom_condition], dim=layer)
    psi_raw = psi_raw.chunk({layer: len(psi_raw[layer])})  # need to rechunk to new size

    # rename to correct dimension and add correct vertical coordinate
    psi = psi_raw.rename({layer: interface})
    psi[interface] = ds[interface]
    psi.name = "psi"  # set variable name in dataarray

    # convert kg.s-1 to Sv (1e6 m3.s-1)
    psi_Sv = psi / rho0 / 1.0e6
    # optionally add offset to make plots cleaner
    if add_offset:
        psi_Sv += offset
    return psi_Sv

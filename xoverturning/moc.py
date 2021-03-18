import xarray as xr
from xoverturning.compfunc import (
    substract_hml,
    rotate_velocities_to_geo,
    interp_to_grid_center,
    select_basins,
    compute_streamfunction,
)


def calcmoc(
    ds,
    basin="global",
    rotate=False,
    remove_hml=False,
    add_offset=False,
    offset=0.1,
    rho0=1035.0,
    zonaldim="xh",
    layer="z_l",
    interface="z_i",
    truelon="geolon",
    truelat="geolat",
    landmask="wet",
    umo="umo",
    vmo="vmo",
    uhml="uhml",
    vhml="vhml",
):
    """Compute Meridional Overturning

    Args:
        ds (xarray.Dataset): input dataset. It should contain at least
                             umo, vmo and some grid information
        basin (str, optional): Basin to use (global/atl-arc/indopac). Defaults to "global".
        rotate (bool, optional): Rotate velocities to true North. Defaults to False.
        remove_hml (bool, optional): Substract Thickness Flux to Restratify Mixed Layer.
                                     Defaults to False.
        add_offset (bool, optional): Add offset to clean up zero contours in plot. Defaults to False.
        offset (float, optional): offset for contours, should be small. Defaults to 0.1.
        rho0 (float, optional): Average density of seawater. Defaults to 1035.0.
        zonaldim (str, optional): name of zonal dimension (for streamfunction).
                                  Defaults to "xh".
        vertdim (str, optional): Vertical dimension for streamfunction integration
                                 (e.g. z_l, rho2_l). Defaults to "z_l".
        truelon (str, optional): name of lon variable. Only used if generating basin codes
                                 (i.e. basin does not exist in dataset). Defaults to "geolon".
        truelat (str, optional): name of lat variables. Same reason as truelon. Defaults to "geolat".
        landmask (str, optional): name of land sea mask variable, also for basin codes. Defaults to "wet".
        umo (str, optional): override for transport name. Defaults to "umo".
        vmo (str, optional): override for transport name. Defaults to "vmo".
        uhml (str, optional): overide for thickness flux. Defaults to "uhml".
        vhml (str, optional): override for thickness flux. Defaults to "vhml".

    Returns:
        xarray.DataArray: meridional overturning
    """

    if remove_hml:
        ucorr, vcorr = substract_hml(ds, umo=umo, vmo=vmo, uhml=uhml, vhml=vhml)
    else:
        ucorr, vcorr = ds[umo], ds[vmo]

    if rotate:
        u_ctr, v_ctr = rotate_velocities_to_geo(ds, ucorr, vcorr)
    else:
        u_ctr, v_ctr = interp_to_grid_center(ds, ucorr, vcorr)

    maskmoc = select_basins(ds, basin=basin, lon=truelon, lat=truelat, mask=landmask)

    ds_v = xr.Dataset()
    ds_v['v'] = v_ctr.where(maskmoc)
    for var in ['xh', 'yh', 'xq', 'yq', layer, interface]:
        ds_v[var] = ds[var]

    moc = compute_streamfunction(
        ds_v,
        xdim=zonaldim,
        layer=layer,
        interface=interface,
        rho0=rho0,
        add_offset=add_offset,
        offset=offset,
    )

    return moc

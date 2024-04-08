import tifffile
from scipy.ndimage import median_filter
import xarray as xr
import numpy as np


def integrator(
    ai,
    tiff_file=None,
    nc_file=None,
    mask=None,
    radial_range=[0.1, 10.1],
    delta_q=0.0025,
    median_filter_size=2,
    plot=True,
    plot_range=None,
    update_ds=False,
):
    if tiff_file is not None:
        img = tifffile.imread(tiff_file)
        if median_filter_size > 1:
            img = median_filter(img, size=median_filter_size)
        ds = xr.Dataset()
        str_ax_title = tiff_file

    if nc_file is not None:
        with xr.open_dataset(nc_file) as ds:
            for k in [
                "i1d",
                "i2d",
                "radial",
                "azimuthal",
                "Y_obs",
                "Y_calc",
                "Y_bkg_auto",
                "Y_bkg_gsas",
                "Y_bkg",
                "Y_bkg_arpls",
                "X_in_q",
                "X_in_d",
                "X_in_tth",
            ]:
                if k in ds.keys():
                    del ds[k]
            da_dark = ds.dexela_img_dark
            dark_from = nc_file
            ds.attrs["dark_from"] = dark_from
            img = (ds.dexela_img.astype("float32") - da_dark.astype("float32")).values
            if median_filter_size > 1:
                img = median_filter(img, size=median_filter_size)
        str_ax_title = tiff_file

    npt = int(np.ceil((radial_range[1] - radial_range[0]) / delta_q))
    radial_range = [radial_range[0], radial_range[0] + delta_q * npt]

    # integrate
    i2d = ai.integrate2d(
        data=img,
        npt_rad=npt,
        npt_azim=360,
        filename=None,
        correctSolidAngle=True,
        variance=None,
        error_model=None,
        radial_range=radial_range,
        azimuth_range=None,
        mask=mask,
        dummy=np.NaN,
        delta_dummy=None,
        polarization_factor=None,
        dark=None,
        flat=None,
        method="bbox",
        unit="q_A^-1",
        safe=True,
        normalization_factor=1.0,
        metadata=None,
    )

    # ceate new data arrays
    try:
        radial_unit = str(i2d.unit)
        xlabel = str(i2d.unit.label)
    except:
        radial_unit = str(i2d.unit[0])
        xlabel = str(i2d.unit[0].label)

    ds["i2d"] = xr.DataArray(
        data=i2d.intensity.astype("float32"),
        coords=[i2d.azimuthal.astype("float32"), i2d.radial.astype("float32")],
        dims=["azimuthal", "radial"],
        attrs={
            "radial_unit": radial_unit,
            "xlabel": xlabel,
            "ylabel": r"Azimuthal angle $\chi$ ($^{o}$)",
        },
    )

    ds.attrs = ds.attrs | ai_dict

    ds.attrs["median_filter_size"] = median_filter_size
    ds.attrs["delta_q"] = delta_q
    ds.attrs["radial_range"] = radial_range

    if update_ds:
        if nc_file is not None:
            ds.to_netcdf(nc_file + ".new.nc", engine="scipy")
            time.sleep(0.1)
            shutil.move(nc_file + ".new.nc", nc_file)
        else:
            ds.to_netcdf(tiff_file + ".nc", engine="scipy")

    if plot:
        fig = plt.figure()

        if plot_range is None:
            plot_range = radial_range

        ax = fig.add_subplot(2, 1, 1)
        da_i2d = ds.i2d
        np.log(da_i2d).plot.imshow(
            ax=ax,
            robust=True,
            add_colorbar=False,
            cmap="Greys",
        )
        ax.set_ylabel(ds.i2d.ylabel)
        ax.set_xlabel(None)
        ax.set_xticks([])
        ax.set_title(str_ax_title, fontsize=7)
        ax.set_xlim(plot_range)

        ax = fig.add_subplot(2, 1, 2)

        da_i2d.mean(dim="azimuthal").plot()
        ax.set_xlabel(ds.i2d.xlabel)
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_yscale("log")
        ax.set_xlim(plot_range)

    return ds

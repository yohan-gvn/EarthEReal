"""
Streams any CMR-discovered atmosphere/ocean variable over OPeNDAP with the
user's Earthdata token. Only the requested box, period and level are
transferred; nothing is written to disk.

NOT TESTED LIVE (no network access or token where this was written). First
run: add one monthly product and analyse one month.
"""

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import xarray as xr

from . import cmr, app_config as config
from .grid import (LAT_NAMES, LEV_NAMES, LON_NAMES, TIME_NAMES,
                   contiguous_runs, find_name, lat_slice, lon_indices,
                   nearest_index, valid_collection_id, valid_variable_name)


# --------------------------------------------------------------------------
# Authenticated session (keeps the bearer token across Earthdata redirects)
# --------------------------------------------------------------------------

class EarthdataSession(requests.Session):
    def __init__(self, token):
        super().__init__()
        self.headers["Authorization"] = f"Bearer {token}"

    def rebuild_auth(self, prepared_request, response):
        host = urlparse(prepared_request.url).hostname or ""
        if host.endswith(config.TRUSTED_AUTH_SUFFIXES):
            prepared_request.headers["Authorization"] = (
                self.headers["Authorization"])
        else:
            prepared_request.headers.pop("Authorization", None)


def _need_token(token):
    if not token:
        raise PermissionError(
            "No Earthdata token. Send it in the X-Earthdata-Token header or "
            "set EARTHDATA_TOKEN.")


# --------------------------------------------------------------------------
# Product descriptors (what "Add product" returns and /analyze accepts)
# --------------------------------------------------------------------------

def normalise_descriptor(d):
    """Validate a client-supplied product descriptor. The client is not
    trusted: every field is re-checked here."""
    if not isinstance(d, dict):
        raise ValueError("Each product must be an object.")
    cid = d.get("collection")
    if not valid_collection_id(cid):
        raise ValueError(
            "Invalid collection id. Add the product first (POST /api/products/add/) and send back what it returns.")
    vars_ = d.get("variables") or ([d["variable"]] if d.get("variable") else [])
    if not (1 <= len(vars_) <= 2) or not all(map(valid_variable_name, vars_)):
        raise ValueError("Give one variable, or two for a derived magnitude.")
    derive = d.get("derive")
    if len(vars_) == 2 and derive != "magnitude":
        raise ValueError("Two variables require derive='magnitude'.")
    if len(vars_) == 1 and derive:
        raise ValueError("'derive' needs two variables.")
    level = d.get("level")
    if level is not None:
        level = float(level)
    label = str(d.get("label") or "+".join(vars_))[:120]
    return {
        "id": f"{cid}|{'+'.join(vars_)}|{derive or 'none'}|{level}",
        "collection": cid, "variables": list(vars_), "derive": derive,
        "level": level, "label": label,
    }


# --------------------------------------------------------------------------
# Reading one remote file
# --------------------------------------------------------------------------

def _open(session, url):
    try:
        return xr.open_dataset(
            url, engine="pydap", session=session, timeout=config.HTTP_TIMEOUT)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open {url}. Check the token and that you have "
            f"approved the data provider in your Earthdata profile. "
            f"({type(exc).__name__}: {exc})") from exc


def describe_variables(session, url):
    """Gridded variables in a sample file, with dims, units, level range."""
    out = []
    with _open(session, url) as ds:
        lat = find_name(ds.variables, LAT_NAMES)
        lon = find_name(ds.variables, LON_NAMES)
        lev = find_name(ds.dims, LEV_NAMES)
        for name, v in ds.data_vars.items():
            if lat in v.dims and lon in v.dims:
                item = {"variable": name,
                        "long_name": v.attrs.get("long_name"),
                        "units": v.attrs.get("units"),
                        "dims": list(v.dims),
                        "has_levels": bool(lev and lev in v.dims)}
                if item["has_levels"]:
                    lv = ds[lev].values
                    item["level_range"] = [float(np.min(lv)),
                                           float(np.max(lv))]
                    item["level_units"] = ds[lev].attrs.get("units")
                out.append(item)
    return out


def _subset(ds, desc, box, t0, t1):
    names = list(ds.variables)
    lat_n = find_name(names, LAT_NAMES)
    lon_n = find_name(names, LON_NAMES)
    time_n = find_name(names, TIME_NAMES)
    lev_n = find_name(ds.dims, LEV_NAMES)
    if not (lat_n and lon_n):
        raise ValueError("File has no recognisable lat/lon coordinates.")

    missing = [v for v in desc["variables"] if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables not in file: {missing}")
    sub = ds[desc["variables"]]

    indexers = {}
    ls = lat_slice(ds[lat_n].values, box[0], box[1])
    if ls is None:
        raise ValueError("The selected box contains no grid points.")
    indexers[lat_n] = ls

    if time_n and time_n in sub.dims:
        tv = pd.to_datetime(ds[time_n].values)
        keep = np.where((tv >= t0) & (tv <= t1))[0]
        if len(keep) == 0:
            return None
        indexers[time_n] = slice(int(keep[0]), int(keep[-1]) + 1)

    has_lev = bool(lev_n and any(lev_n in sub[v].dims
                                 for v in desc["variables"]))
    if has_lev:
        if desc["level"] is None:
            raise ValueError(
                f"{desc['variables']} has vertical levels ({lev_n}); set "
                f"'level' on the product.")
        indexers[lev_n] = nearest_index(ds[lev_n].values, desc["level"])

    lon_idx = lon_indices(ds[lon_n].values, box[2], box[3])
    if len(lon_idx) == 0:
        raise ValueError("The selected box contains no grid points.")

    parts = []
    for a, b in contiguous_runs(lon_idx):
        idx = dict(indexers)
        idx[lon_n] = slice(a, b)
        parts.append(sub.isel(idx).load())
    out = xr.concat(parts, dim=lon_n) if len(parts) > 1 else parts[0]

    rename = {lat_n: "lat", lon_n: "lon"}
    if time_n:
        rename[time_n] = "time"
    out = out.rename({k: v for k, v in rename.items() if k in out.dims
                      or k in out.coords})
    out = out.assign_coords(lon=out["lon"] % 360).sortby("lon").sortby("lat")

    if desc["derive"] == "magnitude":
        a, b = desc["variables"]
        da = np.sqrt(out[a] ** 2 + out[b] ** 2)
        da.attrs["units"] = out[a].attrs.get("units", "")
    else:
        da = out[desc["variables"][0]]
    if "time" not in da.dims:
        da = da.expand_dims(time=[pd.Timestamp(t0)])
    return da


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def list_variables(token, collection_id):
    """What the user can pick after adding a collection: read from a sample
    file when possible (authoritative), else from CMR's records."""
    if not valid_collection_id(collection_id):
        raise ValueError("Invalid collection id.")
    if token:
        try:
            return describe_variables(EarthdataSession(token),
                                      cmr.sample_granule_url(collection_id))
        except Exception:
            pass
    return cmr.collection_variables(collection_id)


def add_product(token, descriptor):
    """'Add product': validate and return a canonical descriptor the client
    keeps and sends back to /analyze. Confirms the variable really exists
    when a sample file can be read."""
    desc = normalise_descriptor(descriptor)
    info = cmr.collection_info(desc["collection"])
    known = list_variables(token, desc["collection"])
    names = {v["variable"] for v in known}
    if names and not set(desc["variables"]) <= names:
        raise ValueError(
            f"{desc['variables']} not found in {info['short_name']}.")
    meta = next((v for v in known if v["variable"] == desc["variables"][0]),
                {})
    if descriptor.get("label") is None:
        desc["label"] = (f"{meta.get('long_name') or desc['variables'][0]} "
                         f"({info['short_name']})")[:120]
    desc["units"] = meta.get("units")
    desc["has_levels"] = meta.get("has_levels", False)
    desc["collection_info"] = info
    return desc


def load_product(token, descriptor, start, end, bbox):
    """DataArray (time, lat, lon) of monthly values for the box, streamed."""
    _need_token(token)
    desc = normalise_descriptor(descriptor)
    sp, ep = pd.Period(start, "M"), pd.Period(end, "M")
    t0, t1 = sp.start_time, ep.end_time
    box = (bbox["lat_min"], bbox["lat_max"], bbox["lon_min"], bbox["lon_max"])

    urls = cmr.granule_urls(desc["collection"], sp, ep)

    def work(url):
        return _subset_file(token, url, desc, box, t0, t1)

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        arrays = [a for a in pool.map(work, urls) if a is not None]
    if not arrays:
        raise ValueError("No data fell inside the requested period.")

    da = xr.concat(arrays, dim="time").sortby("time")
    da = da.sel(time=slice(t0, t1))

    # Everything is analysed as monthly means. Daily/hourly products are
    # averaged to months (data volume is capped by MAX_GRANULES).
    times = pd.DatetimeIndex(da["time"].values)
    step = np.median(np.diff(times.values).astype("timedelta64[h]").astype(float)) \
        if len(times) > 1 else 24 * 30
    resampled = False
    if step < 24 * 25:
        da = da.resample(time="MS").mean()
        resampled = True
    da = da.assign_coords(
        time=pd.DatetimeIndex(da["time"].values).to_period("M").to_timestamp())
    da.attrs["resampled_to_monthly"] = resampled
    da.name = desc["id"]
    return da


def _subset_file(token, url, desc, box, t0, t1):
    session = EarthdataSession(token)
    with _open(session, url) as ds:
        return _subset(ds, desc, box, t0, t1)


def point_timeseries(token, data_id, lat, lon, start, end):
    """Single-point series via Giovanni's REST endpoint (parameter format
    from NASA forum examples; verify before relying on it)."""
    r = requests.get(
        config.GIOVANNI_TIMESERIES_URL,
        params={"data": data_id,
                "location": f"[{round(lat, 4)},{round(lon, 4)}]",
                "time": f"{start}/{end}"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

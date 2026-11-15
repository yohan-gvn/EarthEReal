"""EarthEReal - Earth & Climate Explorer (Streamlit front end for the Django
backend). Every number shown comes from the backend; AI text is labelled."""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timezone
from html import escape

import pandas as pd
import streamlit as st

import charts
import chat_context
import workflow
from api_client import ApiError, BackendClient, default_base_url
from catalog import TABS
from catalog.base import bbox_overlaps, estimate_files

MAX_GRANULES = int(os.environ.get("EARTHEREAL_MAX_GRANULES", "1500"))   # mirror of backend config
MIN_MONTHS = 8                                                           # backend trend tests need >= 8
PRESETS = {
    "Custom": None,
    "Philippines region": (4.0, 22.0, 116.0, 128.0),
    "Western Pacific": (-10.0, 40.0, 100.0, 180.0),
    "Contiguous US": (25.0, 50.0, -125.0, -67.0),
    "Global": (-90.0, 90.0, -180.0, 180.0),
}

st.set_page_config(page_title="EarthEReal | Earth & Climate Explorer", page_icon="🌍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #07111f; color: #e5edf7; }
[data-testid="stSidebar"] { background-color: #0c1929; border-right: 1px solid #1b3045; }
[data-testid="stHeader"] { background-color: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1700px; }

/* ---------- Readable text everywhere ---------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] *,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
  color: #e5edf7;
}

/* Secondary / muted text */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption, small { color: #8ca3bb !important; }

/* Metrics (st.metric) */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #8ca3bb !important; }
[data-testid="stMetricValue"] { color: #e5edf7 !important; }

/* Inputs, selects, multiselects */
input, textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
  background-color: #10253a !important;
  color: #e5edf7 !important;
  border-color: #1d3a50 !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] div { color: #e5edf7 !important; }
[data-baseweb="select"] svg { fill: #8ca3bb !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background-color: #0c1c2e !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: #e5edf7 !important; }
[data-baseweb="popover"] li:hover { background-color: #173b40 !important; }
[data-baseweb="tag"] { background-color: #173b40 !important; }
[data-baseweb="tag"] span { color: #48e0cf !important; }
input::placeholder, textarea::placeholder { color: #6b829a !important; }

/* Tabs */
[data-baseweb="tab-list"] button p { color: #8ca3bb !important; }
[data-baseweb="tab-list"] button[aria-selected="true"] p { color: #22d3c5 !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  background-color: #10253a; color: #e5edf7; border: 1px solid #1d3a50;
}
.stButton > button:hover { border-color: #22d3c5; color: #22d3c5; }

/* Dataframes / tables */
[data-testid="stDataFrame"], [data-testid="stTable"] table, [data-testid="stTable"] td, [data-testid="stTable"] th {
  color: #e5edf7 !important;
}

/* Slider values */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] div[data-baseweb="slider"] div { color: #e5edf7 !important; }

/* Your custom classes (unchanged) */
.brand { font-size: 28px; font-weight: 800; letter-spacing: -1px; color: #e5edf7; }
.brand span { color: #22d3c5; }
.subtitle { color: #8ca3bb; font-size: 13px; margin-top: 3px; }
.section-title { font-size: 19px; font-weight: 700; color: #e5edf7; margin-bottom: 5px; }
.section-description { color: #8ca3bb; font-size: 12px; margin-bottom: 15px; }
.metric-card { background: linear-gradient(145deg, #10253a, #0c1c2e); border: 1px solid #1d3a50;
  border-radius: 12px; padding: 16px; min-height: 120px; }
.metric-label { color: #8ca3bb; font-size: 12px; font-weight: 500; }
.metric-value { color: #e5edf7; font-size: 24px; font-weight: 700; margin-top: 8px; }
.metric-caption { color: #22d3c5; font-size: 11px; margin-top: 3px; }
.status-pill { background-color: #173b40; color: #48e0cf; border: 1px solid #23615f;
  padding: 5px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.status-pill.bad { background-color: #3f1d24; color: #ff8a9b; border-color: #6b2a35; }
.tag-nasa { color: #22d3c5; font-size: 11px; font-weight: 600; }
.tag-ai { color: #f59e0b; font-size: 11px; font-weight: 600; }
.info-panel { background-color: #0c1c2e; border: 1px solid #1d3a50; border-radius: 12px; padding: 14px; }
hr { border-color: #1b3045; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached calls
@st.cache_data(ttl=20, show_spinner=False)
def cached_health(base_url: str, token: str):
    return BackendClient(base_url, token).health()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_listing(base_url: str, token: str, short_name: str):
    return BackendClient(base_url, token).collection_listing(short_name)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_add(base_url: str, token: str, body: dict):
    return BackendClient(base_url, token).add_product(body)


def section_header(title, description=None):
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="section-description">{escape(description)}</div>',
                    unsafe_allow_html=True)


def metric_card(label, value, caption):
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{escape(str(label))}</div>
<div class="metric-value">{escape(str(value))}</div>
<div class="metric-caption">{escape(str(caption))}</div></div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">EARTH<span>EREAL</span></div>'
                '<div class="subtitle">Earth & Climate Explorer</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Backend")
    base_url = st.text_input("Backend URL", value=default_base_url(),
                             help="Django server. Default comes from EARTHEREAL_API_URL.").strip().rstrip("/")
    user_token = st.text_input("Earthdata token (optional)", type="password",
                               help="Leave empty to use EARTHDATA_TOKEN configured on the server. "
                                    "It is sent only to your backend.")
    client = BackendClient(base_url, user_token)
    health, health_error = None, None
    try:
        health = cached_health(base_url, user_token.strip())
    except ApiError as exc:
        health_error = str(exc)
    if health:
        st.markdown('<span class="status-pill">● BACKEND ONLINE</span>', unsafe_allow_html=True)
        token_ok = bool(health.get("earthdata_token_set")) or bool(user_token.strip())
        st.caption("Earthdata token: " + ("available" if token_ok else "MISSING (analysis needs one)"))
        chat_ok = bool(health.get("chat_enabled", health.get("literature_enabled")))
        st.caption("AI assistant: " + ("enabled" if chat_ok else "disabled (no ANTHROPIC_API_KEY on server)"))
        st.caption("Archives: " + ", ".join(health.get("providers", [])))
        max_products = int(health.get("max_products", 3))
    else:
        st.markdown('<span class="status-pill bad">● BACKEND OFFLINE</span>', unsafe_allow_html=True)
        st.caption(health_error)
        token_ok, chat_ok, max_products = False, False, 3

    st.markdown("---")
    st.markdown("### Region")
    preset = st.selectbox("Preset", list(PRESETS), help="Presets are only bounding boxes; edit freely.")
    base = PRESETS[preset] or (0.0, 30.0, 110.0, 150.0)
    c1, c2 = st.columns(2)
    lat_min = c1.number_input("Lat min", -90.0, 90.0, base[0], 0.5, key=f"latmin_{preset}")
    lat_max = c2.number_input("Lat max", -90.0, 90.0, base[1], 0.5, key=f"latmax_{preset}")
    lon_min = c1.number_input("Lon min", -180.0, 360.0, base[2], 0.5, key=f"lonmin_{preset}")
    lon_max = c2.number_input("Lon max", -180.0, 360.0, base[3], 0.5, key=f"lonmax_{preset}")
    bbox = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}

    st.markdown("### Period")
    start_d = st.date_input("Start month", value=date(2017, 1, 1), min_value=date(1980, 1, 1),
                            max_value=date.today())
    end_d = st.date_input("End month", value=date(2019, 12, 1), min_value=date(1980, 1, 1),
                          max_value=date.today())
    start, end = start_d.strftime("%Y-%m"), end_d.strftime("%Y-%m")
    n_months = (end_d.year - start_d.year) * 12 + end_d.month - start_d.month + 1
    st.caption(f"{start} to {end}: {max(n_months, 0)} months (only year and month are used)")
    alpha = st.selectbox("Significance level (alpha)", [0.05, 0.01, 0.10], index=0)


# ------------------------------------------------------------------ header
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown('<div class="brand">EARTH<span>EREAL</span></div>'
                '<div class="subtitle">NASA Earth-science analysis: pick variables, a region and a period; '
                'every value comes from the backend, never from this page.</div>', unsafe_allow_html=True)
st.markdown("---")

main_col, chat_col = st.columns([2.4, 1.1], gap="large")


# ------------------------------------------------------------------ selection
def render_selection(tab, groups):
    selections, summary = {}, []
    amb = st.session_state.get("ambiguous", {}).get(tab, {})
    for gi, g in enumerate(groups):
        with st.expander(f"{g.label}  ·  {g.short_name} v{g.version}", expanded=(gi == 0)):
            st.caption(f"{g.product} · {g.temporal} granules")
            if g.note:
                st.caption(g.note)
            links = " · ".join(f"[{n}]({u})" for n, u in
                               (("Dataset", g.source_url), ("Variable glossary", g.glossary_url)) if u)
            if links:
                st.markdown(links)
            keys = st.multiselect("Variables", [o.key for o in g.options],
                                  format_func=lambda k, g=g: g.option(k).label, key=f"opt_{g.key}")
            for k in keys:
                if g.option(k).description:
                    st.caption(f"{g.option(k).label}: {g.option(k).description}")
            levels = []
            if g.levels and keys:
                levels = st.multiselect(f"Pressure levels ({g.level_unit})", list(g.levels),
                                        default=list(g.default_levels),
                                        format_func=lambda v: f"{v:g}", key=f"lvl_{g.key}")
            choices = {}
            for ak, cands in amb.items():
                gk, ok = ak.split("|")
                if gk != g.key:
                    continue
                names = [c["variable"] for c in cands]
                choices[ok] = st.selectbox(
                    f"Several variables match '{g.option(ok).label}': choose one", names,
                    format_func=lambda n, cands=cands: next(
                        f"{c['variable']} - {c.get('long_name') or ''} ({c.get('units') or ''})"
                        for c in cands if c["variable"] == n),
                    key=f"var_{g.key}_{ok}")
            if keys:
                selections[g.key] = {"options": keys, "levels": levels, "choices": choices}
                summary.append({"dataset": g.short_name, "version": g.version, "product": g.product,
                                "variables": [g.option(k).label for k in keys],
                                "levels_hPa": levels or None})
    return selections, summary


def validate(groups, selections):
    errors, warnings = [], []
    if not selections:
        errors.append("Select at least one variable.")
    if not (-90 <= lat_min < lat_max <= 90):
        errors.append("Latitude: min must be below max, within -90 to 90.")
    if not (-180 <= lon_min < lon_max <= 360 and lon_max - lon_min <= 360):
        errors.append("Longitude: min must be below max, within -180 to 360, at most 360 apart.")
    if n_months < 1:
        errors.append("The start month is after the end month.")
    elif n_months < MIN_MONTHS:
        errors.append(f"Choose at least {MIN_MONTHS} months: the backend's trend tests need that many points.")
    elif n_months < 36:
        warnings.append("Periods under about 3 years give weak anomaly and trend statistics "
                        "(with under 2 years the deseasonalised anomalies are degenerate).")
    if health is None:
        errors.append("The backend is offline.")
    elif not token_ok:
        errors.append("No Earthdata token: set EARTHDATA_TOKEN on the server or enter one in the sidebar.")
    n_prod = 0
    for g in groups:
        sel = selections.get(g.key)
        if not sel:
            continue
        n_prod += len(sel["options"]) * max(1, len(sel["levels"]))
        files = estimate_files(g, max(n_months, 0))
        if files > MAX_GRANULES:
            fit = int(MAX_GRANULES // g.files_per_month)
            errors.append(f"{g.short_name}: about {files} files for this period, over the backend cap of "
                          f"{MAX_GRANULES} per product. Use at most ~{fit} months.")
        elif files > MAX_GRANULES * 0.6:
            warnings.append(f"{g.short_name}: about {files} files per product; this may take several minutes.")
        if g.coverage and not bbox_overlaps(bbox, g.coverage):
            errors.append(f"{g.short_name}: its domain (contiguous US) does not overlap the selected region.")
        elif g.coverage:
            warnings.append(f"{g.short_name}: data exist only inside the contiguous US part of your region.")
    if n_prod > max_products:
        warnings.append(f"{n_prod} products will run in {math.ceil(n_prod / max_products)} backend requests "
                        f"(limit {max_products} per request); correlations are only computed within a request.")
    return errors, warnings


def signature(tab, selections):
    return json.dumps({"tab": tab, "sel": selections, "bbox": bbox, "start": start, "end": end,
                       "alpha": alpha}, sort_keys=True, default=str)


def run_flow(tab, groups, selections):
    with st.status("Running analysis...", expanded=True) as status:
        st.write("Resolving datasets and variables through the backend...")
        tok = user_token.strip()
        prep = workflow.prepare(groups, selections,
                                lambda s: cached_listing(base_url, tok, s),
                                lambda b: cached_add(base_url, tok, b))
        for w in prep.warnings:
            st.warning(w)
        if prep.ambiguous:
            st.session_state.setdefault("ambiguous", {})[tab] = {
                f"{gk}|{ok}": c for (gk, ok), c in prep.ambiguous.items()}
            status.update(label="Choose which variable to use, then run again.", state="error")
            st.rerun()
        st.session_state.setdefault("ambiguous", {}).pop(tab, None)
        for e in prep.errors:
            (st.error if not prep.products else st.warning)(e)
        if not prep.products:
            status.update(label="Could not prepare the request.", state="error")
            return

        def progress(i, n, labels):
            st.write(f"Request {i}/{n}: streaming NASA data for {', '.join(labels)} "
                     "(this can take several minutes)...")
        merged, errors = workflow.run_analysis(client, prep.products, bbox, start, end, alpha,
                                               max_products, progress)
        if not merged["products"]:
            for e in errors:
                st.error(f"{', '.join(e['products'])}: {e['error']}")
            status.update(label="The backend returned no results.", state="error")
            return
        st.session_state.setdefault("analyses", {})[tab] = {
            "results": merged, "errors": errors, "prepared": prep.products,
            "signature": signature(tab, selections),
            "ran_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
        st.session_state.get("literature", {}).pop(tab, None)
        status.update(label=f"Done: {len(merged['products'])} product(s) analysed.", state="complete")


# ------------------------------------------------------------------ results
def render_results(tab, analysis, stale):
    res, products = analysis["results"], analysis["results"]["products"]
    titles = {p["product"]: p["title"] for p in products}
    st.markdown(f'<span class="tag-nasa">NASA DATA via backend · analysed {analysis["ran_at"]}</span>',
                unsafe_allow_html=True)
    if stale:
        st.warning("The selection, region or period changed since this analysis. The results below "
                   "describe the earlier selection; run again to update.")
    if any(p.get("resampled_to_monthly") for p in products):
        st.info("Daily or hourly data were averaged to monthly means by the backend before analysis.")
    for e in analysis.get("errors", []):
        st.error(f"Failed: {', '.join(e['products'])} - {e['error']}")

    cols = st.columns(min(4, len(products)))
    for i, p in enumerate(products):
        with cols[i % len(cols)]:
            m, o = p["statistics"]["mean"], p["statistics"]["trend_ols"]
            metric_card(p["title"][:48], f"{m['mean']:.4g} {p.get('units') or ''}",
                        f"trend {o['slope_per_decade']:+.3g}/decade (p={o['p_value']:.3f})")
    st.markdown("")

    t_series, t_maps, t_stats, t_cmp, t_lit = st.tabs(
        ["Time series", "Maps", "Statistics", "Compare", "Literature"])
    with t_series:
        kind = st.radio("Show", ["Area mean", "Deseasonalised anomaly"], horizontal=True, key=f"kind_{tab}")
        st.plotly_chart(charts.series_figure(products, "value" if kind == "Area mean" else "anomaly"),
                        width="stretch", key=f"ts_{tab}")
        rows = [{"product": p["title"], "month": t, "value": v, "anomaly": a, "units": p.get("units")}
                for p in products for t, v, a in zip(p["area_mean"]["time"], p["area_mean"]["value"],
                                                     p["area_mean"]["anomaly"])]
        st.download_button("Download area-mean series (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           file_name=f"earthereal_{tab.lower()}_area_mean.csv", mime="text/csv")
    with t_maps:
        pid = st.selectbox("Product", list(titles), format_func=titles.get, key=f"mapsel_{tab}")
        p = next(x for x in products if x["product"] == pid)
        left, right = st.columns(2)
        with left:
            if charts.has_values(p.get("mean_map"), "value"):
                st.plotly_chart(charts.mean_map_figure(p), width="stretch", key=f"mm_{tab}_{pid}")
            else:
                st.info("No valid grid values for the period mean (for example where the variable is undefined).")
        with right:
            if charts.has_values(p.get("trend_map"), "slope_per_decade"):
                st.plotly_chart(charts.trend_map_figure(p), width="stretch", key=f"tm_{tab}_{pid}")
            else:
                st.info("No trend map: trends need a value at every month for a grid point.")
        st.caption(p["trend_map"].get("note", ""))
    with t_stats:
        st.dataframe(charts.stats_dataframe(products), width="stretch", hide_index=True)
        corr = res.get("cross_product_correlation") or []
        if corr:
            st.markdown("**Correlation of anomalies between products**")
            st.dataframe(pd.DataFrame([{"A": titles.get(c["a"], c["a"]), "B": titles.get(c["b"], c["b"]),
                                        "r": round(c["r"], 3), "p": round(c["p_value"], 4),
                                        "n": c["n"], "significant": c["significant"]} for c in corr]),
                         width="stretch", hide_index=True)
        st.caption("Significance is not physical importance. Trends are on deseasonalised monthly anomalies, "
                   "adjusted for autocorrelation.")
    with t_cmp:
        prof = charts.profile_figure(products)
        if prof:
            st.plotly_chart(prof, width="stretch", key=f"prof_{tab}")
        else:
            st.caption("A vertical profile appears when you analyse several levels of one variable "
                       "(or several geopotential-height levels).")
        if len(products) >= 2:
            ca, cb = st.columns(2)
            a = ca.selectbox("A", list(titles), format_func=titles.get, key=f"da_{tab}")
            b = cb.selectbox("B", list(titles), index=1, format_func=titles.get, key=f"db_{tab}")
            pa = next(x for x in products if x["product"] == a)
            pb = next(x for x in products if x["product"] == b)
            if a == b:
                st.caption("Pick two different products to see their difference.")
            elif (pa.get("units") or "") != (pb.get("units") or ""):
                st.info("These products have different units, so a difference is not meaningful.")
            else:
                st.plotly_chart(charts.difference_figure(pa, pb), width="stretch", key=f"diff_{tab}")
    with t_lit:
        st.caption("Literature search is run by the backend; the written analysis is AI-generated from "
                   "retrieved titles and abstracts, not full papers.")
        q = st.text_input("Question (optional)", key=f"litq_{tab}")
        if st.button("Find related literature", key=f"litbtn_{tab}"):
            with st.spinner("Searching papers and writing the analysis..."):
                try:
                    ctx = chat_context.literature_context(res, res["request"])
                    st.session_state.setdefault("literature", {})[tab] = client.literature(ctx, q)
                except ApiError as exc:
                    st.error(str(exc))
        lit = st.session_state.get("literature", {}).get(tab)
        if lit:
            if lit.get("analysis"):
                st.markdown('<span class="tag-ai">AI-GENERATED INTERPRETATION</span>', unsafe_allow_html=True)
                st.markdown(lit["analysis"])
            elif lit.get("note"):
                st.info(lit["note"])
            for r in lit.get("references", []) or lit.get("retrieved", [])[:8]:
                st.markdown(f"- [{r.get('id', '')}] [{r.get('title', '')}]({r.get('url', '')}) "
                            f"{r.get('authors_venue_year', '')}")


# ------------------------------------------------------------------ main column
with main_col:
    tab = st.radio("Science category", list(TABS), horizontal=True, key="tab",
                   label_visibility="collapsed")
    groups = TABS[tab]
    section_header(tab, "Select variables, set the region and period in the sidebar, then run the analysis.")
    selections, selected_summary = render_selection(tab, groups)
    errors, warnings = validate(groups, selections)
    for w in warnings:
        st.warning(w)
    for e in errors:
        st.error(e)
    if st.button("Run analysis", type="primary", disabled=bool(errors), key=f"run_{tab}"):
        run_flow(tab, groups, selections)

    analysis = st.session_state.get("analyses", {}).get(tab)
    stale = bool(analysis) and analysis["signature"] != signature(tab, selections)
    if analysis:
        st.markdown("---")
        render_results(tab, analysis, stale)

# ------------------------------------------------------------------ chat column
with chat_col:
    section_header("Science assistant")
    st.markdown('<span class="tag-ai">AI-GENERATED · NOT NASA MEASUREMENTS</span>', unsafe_allow_html=True)
    st.caption("Knows your tab, variables, region, period and latest results. Numbers it quotes come from "
               "your analysis; background science is general knowledge.")
    history = st.session_state.setdefault("chat", [])
    box = st.container(height=560)
    qcols = st.columns(2)
    quick = None
    if qcols[0].button("Explain my results", width="stretch"):
        quick = "Explain my latest results in plain language."
    if qcols[1].button("What do these variables measure?", width="stretch"):
        quick = "What do the selected variables measure and why do they matter?"
    prompt = st.chat_input("Ask about the selected variables or results...") or quick
    if history and st.button("Clear conversation", width="stretch"):
        history.clear()
        st.rerun()

    with box:
        for m in history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        if prompt:
            if not chat_ok:
                st.info("The assistant is unavailable: the backend is offline or has no ANTHROPIC_API_KEY.")
            else:
                history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                other = [t for t in TABS if t != tab and st.session_state.get("analyses", {}).get(t)]
                ctx = chat_context.build_context(
                    tab, selected_summary,
                    {"bbox": bbox, "start": start, "end": end, "alpha": alpha},
                    analysis, stale, bool(other))
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            reply = client.chat(history, ctx)["reply"]
                        except ApiError as exc:
                            reply = None
                            st.error(str(exc))
                    if reply:
                        st.markdown(reply)
                        history.append({"role": "assistant", "content": reply})
                    else:
                        history.pop()   # keep the transcript consistent for a retry

st.markdown("---")
st.caption("Data: NASA GES DISC (MERRA-2, AIRS, NLDAS) via your backend. AI text is generated and may be wrong; "
           "check it against the statistics shown.")

import os
import math
import base64


import io


import hashlib
import re
import json
from pathlib import Path


import requests
import urllib.parse


import streamlit as st


import pandas as pd


import pydeck as pdk
import altair as alt


from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from textwrap import dedent


from datetime import datetime, date, timedelta


from typing import Optional





st.set_page_config(


    page_title="Media Monitoring",


    page_icon="NA",


    layout="wide",


    initial_sidebar_state="collapsed",


)


API = os.getenv("API_URL", "http://backend:8000")

# -------------------------------------------------------------------
# Auth state
# -------------------------------------------------------------------

def _set_query_params(**kwargs):
    try:
        st.experimental_set_query_params(**kwargs)
    except Exception:
        pass

def _get_query_params():
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}


def _clear_query_params():
    try:
        st.query_params.clear()
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params()
    except Exception:
        pass


def clear_auth(clear_all: bool = False):
    st.session_state["access_token"] = None
    st.session_state["user"] = None
    if clear_all:
        _clear_query_params()
    else:
        qp = _get_query_params()
        qp.pop("token", None); qp.pop("user", None); qp.pop("role", None); qp.pop("logout", None)
        _set_query_params(**qp)


def auth_headers() -> dict:
    token = st.session_state.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def auth_query_params() -> dict:
    token = st.session_state.get("access_token")
    user = st.session_state.get("user") or {}
    if not token:
        return {}
    return {
        "token": token,
        "user": user.get("username"),
        "role": user.get("role"),
    }


@lru_cache(maxsize=1)
def _load_login_logo_data_uri() -> str | None:
    candidates = [
        Path(__file__).resolve().parent / "data" / "images" / "Media-Monitor-logo.png",
        Path(__file__).resolve().parent.parent / "data" / "images" / "Media-Monitor-logo.png",
        Path.cwd() / "data" / "images" / "Media-Monitor-logo.png",
        Path("/data/images/Media-Monitor-logo.png"),
    ]
    for path in candidates:
        try:
            if path.is_file():
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:image/png;base64,{encoded}"
        except Exception:
            continue
    return None

# -------------------------------------------------------------------
# Global styles + top bar (mirrors media_monitoring_app layout)
# -------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Poppins:wght@500;600;700&display=swap');
:root{
  --nav-bg: #ecfeff;
  --nav-border: #bae6fd;
  --nav-text: #0f172a;
  --nav-active-bg: #fde68a;
  --nav-active-border: #fbbf24;
  --section-bg: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  --section-border: #e2e8f0;
  --section-shadow: 0 16px 40px rgba(15,23,42,0.07);
}
header[data-testid="stHeader"]{ display:none !important; }
div[data-testid="stToolbar"]{ display:none !important; }
.appview-container .main .block-container{ padding-top: 5.5rem; }
.main .block-container{ padding-top: 5.5rem; }
.topbar, .topbar *{ pointer-events:auto !important; }
.topbar{
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 9999;
  background: linear-gradient(180deg, #eef2ff 0%, #e0e7ff 100%);
  border-bottom: 1px solid #e2e8f0;
}
.topbar-inner{
  max-width: 1200px;
  margin: 0 auto;
  padding: 14px 22px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  flex-wrap: nowrap;
}
.brand{
  display:flex;
  align-items:center;
  gap:12px;
  font-weight: 700;
  font-size: 18px;
  letter-spacing:.2px;
  color: #0f172a;
  font-family: 'Outfit', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
.brand .logo{
  width: 36px; height: 36px;
  border-radius: 10px;
  background: #ecfeff;
  border: 1px solid #cffafe;
  display:flex; align-items:center; justify-content:center;
  font-size: 18px;
  color: #0e7490;
}
.topnav{
  display:flex;
  gap: 8px;
  align-items:center;
  flex-wrap: nowrap;
  margin-left: auto;
  flex: 1 1 auto;
  min-width: 0;
  overflow-x: auto;
  padding-bottom: 2px;
}
.topnav::-webkit-scrollbar{ height: 6px; }
.topnav::-webkit-scrollbar-thumb{
  background: rgba(148, 163, 184, 0.6);
  border-radius: 999px;
}
.topnav a{
  text-decoration:none;
  border:1px solid var(--nav-border) !important;
  background: var(--nav-bg) !important;
  color: var(--nav-text) !important;
  padding: 7px 10px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 13px;
  font-family: 'Poppins', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  transition: all .15s ease;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 100px;
}
.user-menu{
  margin-left: 12px;
  position: relative;
  display: flex;
  align-items: center;
}
.user-trigger{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: #ecfeff;
  border: 1px solid #bae6fd;
  font-weight: 600;
  color: #0f172a;
  font-family: 'Poppins', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  cursor: pointer;
}
.user-avatar{
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #0ea5e9;
  color: #ffffff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
}
.user-name{
  font-size: 13px;
  white-space: nowrap;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.user-caret{
  font-size: 12px;
  opacity: 0.7;
}
.user-dropdown{
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  min-width: 160px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
  padding: 6px;
  display: none;
  z-index: 10000;
}
.user-dropdown a{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  text-decoration: none;
  color: #0f172a;
  font-weight: 600;
}
.user-dropdown a:hover{
  background: #f1f5f9;
}
.user-menu:hover .user-dropdown,
.user-menu:focus-within .user-dropdown{
  display: block;
}
.topnav a:hover{
  background: #e0f2fe !important;
  border-color: #bfdbfe !important;
  color: #0ea5e9 !important;
}
.topnav a.active{
  background: var(--nav-active-bg) !important;
  border-color: var(--nav-active-border) !important;
  color: #92400e !important;
  box-shadow: 0 4px 10px rgba(251,191,36, .35);
}
.section{
  background: var(--section-bg);
  border: 1px solid var(--section-border);
  border-radius: 16px;
  padding: 18px;
  box-shadow: var(--section-shadow);
}
.section + .section{ margin-top: 14px; }
.kpi{
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 2px 4px rgba(15, 23, 42, 0.05);
}
.kpi .kpi-label{
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: #64748b;
  font-weight: 600;
}
.kpi .kpi-value{
  margin-top: 8px;
  font-size: 28px;
  font-weight: 700;
  color: #0f172a;
}
.kpi .kpi-hint{
  margin-top: 6px;
  font-size: 12px;
  color: #94a3b8;
}
.badge{
  display:inline-flex; align-items:center; gap:6px;
  padding: 4px 8px;
  border-radius: 999px;
  background: #ecfeff;
  border: 1px solid #cffafe;
  color: #0e7490;
  font-size: 12px;
}
.news-card{
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  padding: 12px;
  box-shadow: 0 15px 35px rgba(15,23,42,0.04);
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.cat-snapshot .stButton>button{
  width: 100%;
  border-radius: 2px !important;
  padding: 12px 14px;
  min-height: 64px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  font-weight: 700;
}
.party-snapshot .stButton>button{
  width: 100%;
  border-radius: 2px !important;
  padding: 12px 14px;
  min-height: 64px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  font-weight: 700;
}
.news-card__media{
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 12px;
  overflow: hidden;
  background: #f8fafc;
}
.news-card__media img{
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.section,
.kpi,
.news-card,
.badge,
.topnav a,
.stButton>button{
  border-radius: 2px !important;
}
.news-card__body h4{
  font-size: 16px;
  margin: 6px 0;
}
.news-card__body h4 a{
  color: #0f172a;
  text-decoration: none;
}
.news-card__body p{
  color: #475569;
  font-size: 13px;
  min-height: 52px;
}
.news-card__meta{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #94a3b8;
}
.news-card-grid{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.news-card.news-card--compact{
  gap: 6px;
  padding: 10px 12px;
}
.news-card.news-card--compact .news-card__media{ display: none; }
.news-card.news-card--compact .news-card__body p{
  min-height: 0;
  margin-bottom: 4px;
}
.top-news-scroll{
  max-height: 320px;
  overflow-y: auto;
  padding-right: 8px;
  margin-right: -6px;
}
.top-news-scroll::-webkit-scrollbar{ width: 8px; }
.top-news-scroll::-webkit-scrollbar-thumb{
  background: #cbd5e1;
  border-radius: 999px;
}
.footer{
  margin-top: 24px;
  padding: 20px 0 30px;
  text-align: center;
  font-size: 13px;
  color: #475569;
  border-top: 1px dashed #e2e8f0;
}
@media (max-width: 820px){
  .topbar-inner{ padding: 12px 14px; gap: 10px; }
  .topnav a{ padding: 6px 8px; font-size: 12px; flex: 0 0 88px; }
  .user-name{ max-width: 140px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

def get_query_params():


    qp = {}


    try:


        qp = dict(st.query_params)


    except Exception:


        qp = _get_query_params()


    return qp or {}


# Initialize auth state after query helpers are available
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "user" not in st.session_state:
    st.session_state["user"] = None

# Restore auth from query params if present (helps keep login across page reloads)
_qp_init = get_query_params()
if not st.session_state.get("access_token") and _qp_init.get("token"):
    st.session_state["access_token"] = (_qp_init.get("token")[0] if isinstance(_qp_init.get("token"), list) else _qp_init.get("token"))
    st.session_state["user"] = {
        "username": (_qp_init.get("user")[0] if isinstance(_qp_init.get("user"), list) else _qp_init.get("user")),
        "role": (_qp_init.get("role")[0] if isinstance(_qp_init.get("role"), list) else _qp_init.get("role")),
    }








def get_page() -> str:


    qp = get_query_params()


    page = qp.get("page", "Category")


    if isinstance(page, list):


        page = page[0] if page else "Category"


    page = (page or "Category").strip()


    page_lookup = {


        "dashboard": "Dashboard",


        "category": "Category",


        "overview": "Category",


        "party": "Party",


        "candidate": "Candidate",


        "candidates": "Candidate",


        "geography": "Geography",


        "geo": "Geography",


        "media": "Media",
        "admin": "UserManage",
        "usermanage": "UserManage",
        "user manage": "UserManage",


    }


    return page_lookup.get(page.lower(), "Category")








def build_nav_href(page_value: str) -> str:


    params = {"page": page_value, **auth_query_params()}
    return "?" + urllib.parse.urlencode(params)


def build_logout_href(page_value: str) -> str:
    params = {"page": page_value, **auth_query_params(), "logout": "1"}
    return "?" + urllib.parse.urlencode(params)








def render_topbar(
    active_page: str,
    is_admin: bool = False,
    username: Optional[str] = None,
    role: Optional[str] = None,
):


    def cls(name: str) -> str:


        return "active" if active_page == name else ""


    nav_items = [
        ("Category", "Category"),
        ("Party", "Party"),
        ("Candidate", "Candidate"),
        ("Geography", "Geography"),
        ("Media", "Media"),
    ]
    if is_admin:
        nav_items.insert(0, ("Dashboard", "Dashboard"))
    if is_admin:
        nav_items.append(("UserManage", "User Manage"))

    links_html = "\n".join(
        f'<a class="{cls(name)}" href="{build_nav_href(name)}" target="_self">{label}</a>'
        for name, label in nav_items
    )
    user_html = ""
    if username:
        initial = str(username).strip()[:1].upper() or "U"
        role_label = f" ({role})" if role else ""
        signed_in_label = f"{username}"
        logout_href = build_logout_href(active_page)
        user_html = dedent(
            f"""
            <div class="user-menu">
              <div class="user-trigger" tabindex="0" role="button" aria-haspopup="true">
                <div class="user-avatar">{initial}</div>
                <div class="user-name">{signed_in_label}</div>
                <div class="user-caret">&#9662;</div>
              </div>
              <div class="user-dropdown" role="menu">
                <a href="{logout_href}" target="_self">Log out</a>
              </div>
            </div>
            """
        ).strip()

    nav_lines = [
        '<div class="topbar">',
        '<div class="topbar-inner">',
        '<div class="brand">',
        '<div class="logo">MM</div>',
        '<div>Media Monitoring</div>',
        '</div>',
        '<div class="topnav">',
        links_html,
        '</div>',
        user_html,
        '</div>',
        '</div>',
    ]
    nav = "\n".join(line for line in nav_lines if line)

    # Render in-page so global CSS applies.
    st.markdown(nav, unsafe_allow_html=True)


def render_html_block(html: str):
    """Helper to render dedented HTML safely."""
    st.markdown(dedent(html), unsafe_allow_html=True)







# -------------------------------------------------------------------


# Shared UI helpers


# -------------------------------------------------------------------





TIMELINE_OPTIONS = ["Last 24 hours", "This week", "This month", "Custom"]





def _format_custom_date(d: date) -> str:


    return d.strftime("%d-%b-%Y").upper()








def _coerce_custom_range(custom_range):


    if custom_range is None:


        return None


    vals = list(custom_range) if isinstance(custom_range, (list, tuple)) else [custom_range, custom_range]


    def _as_date(val):


        if isinstance(val, datetime):


            return val.date()


        if isinstance(val, date):


            return val


        if hasattr(val, "to_pydatetime"):


            try:


                return val.to_pydatetime().date()


            except Exception:


                return None


        return None


    d0 = _as_date(vals[0]); d1 = _as_date(vals[1] if len(vals) > 1 else vals[0])


    if d0 is None or d1 is None:


        return None


    return datetime.combine(d0, datetime.min.time()), datetime.combine(d1, datetime.max.time())








def _resolve_window(sel: str, custom_range):


    now = datetime.now()


    if sel == "Last 24 hours":


        start = now - timedelta(days=1); end = now


    elif sel == "This week":


        start = datetime.combine(date.today() - timedelta(days=6), datetime.min.time()); end = now


    elif sel == "This month":


        first = date.today().replace(day=1)


        start = datetime.combine(first, datetime.min.time()); end = now


    elif sel == "Custom":


        custom_window = _coerce_custom_range(custom_range)


        if custom_window:


            start, end = custom_window


        else:


            start = now - timedelta(days=7); end = now


    else:


        start = now - timedelta(days=7); end = now


    return start, end








def timeline_control(key_prefix="ov"):


    default_idx = TIMELINE_OPTIONS.index("This week")


    picker_cols = st.columns([2.4, 1.6])


    with picker_cols[0]:


        sel = st.radio(


            "Date range",


            TIMELINE_OPTIONS,


            index=default_idx,


            key=f"{key_prefix}_timeline",


            horizontal=True,


        )


    default_range = (date.today() - timedelta(days=7), date.today())


    with picker_cols[1]:


        custom_range = None


        if sel == "Custom":


            c1, c2 = st.columns(2)


            start_key = f"{key_prefix}_custom_start"


            end_key = f"{key_prefix}_custom_end"


            prev_start = st.session_state.get(start_key, default_range[0])


            prev_end = st.session_state.get(end_key, default_range[1])


            with c1:


                start_val = st.date_input(


                    f"Start date ({_format_custom_date(prev_start)})",


                    value=prev_start,


                    key=start_key,


                    format="YYYY-MM-DD",


                )


            with c2:


                end_val = st.date_input(


                    f"End date ({_format_custom_date(prev_end)})",


                    value=prev_end,


                    key=end_key,


                    format="YYYY-MM-DD",


                )


            custom_range = (start_val, end_val)


        else:


            st.date_input(


                "Custom range",


                value=default_range,


                key=f"{key_prefix}_dates_preview",


                disabled=True,


                format="YYYY-MM-DD",


            )


    start_dt, end_dt = _resolve_window(sel, custom_range if sel == "Custom" else None)


    day_count = max((end_dt.date() - start_dt.date()).days + 1, 1)


    fmt = lambda d: d.strftime("%d-%b-%Y").upper()


    st.caption(f"{fmt(start_dt)} to {fmt(end_dt)} | {day_count} day{'s' if day_count != 1 else ''}")


    return sel, custom_range if sel == "Custom" else None, start_dt, end_dt


def _date_params(start_dt: datetime, end_dt: datetime) -> dict:
    params = {}
    if start_dt:
        params["start_date"] = start_dt.date().isoformat()
    if end_dt:
        params["end_date"] = end_dt.date().isoformat()
    return params

def _format_big_number(value) -> str:


    try:


        return f"{int(value):,}"


    except Exception:


        return "0"








def _news_card_html(row, show_image: bool = False):


    title = row.get("title") or row.get("Headline") or "Untitled"


    link = row.get("url") or row.get("link") or row.get("source_url") or "#"


    desc = row.get("summary") or row.get("description") or ""


    date_str = format_date_pretty(row.get("published_at") or row.get("published"), use_bn=SHOW_BN)


    category = row.get("category") or row.get("Category") or "General"


    portal = row.get("portal") or row.get("source") or ""


    card_classes = "news-card"


    if not show_image:


        card_classes += " news-card--compact"


    media_block = ""


    if show_image:


        img = row.get("image") or row.get("thumbnail") or row.get("img") or "https://placehold.co/320x180?text=News"


        media_block = (


            f'<div class="news-card__media">'


            f'<img src="{img}" alt="{title}" />'


            f"</div>"


        )


    desc_html = f"<p>{desc}</p>" if desc else ""


    meta_right = portal or ""


    parts = [


        f'<div class="{card_classes}">',


        media_block,


        '<div class="news-card__body">',


        f'<span class="badge">{category}</span>',


        f'<h4><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></h4>',


        desc_html,


        '<div class="news-card__meta">',


        f"<span>{date_str}</span>",


        f"<span>{meta_right}</span>",


        "</div>",


        "</div>",


        "</div>",


    ]


    return "\n".join(p for p in parts if p)








def _add_pseudo_latlon(df: pd.DataFrame) -> pd.DataFrame:


    """


    If lat/lon missing, generate deterministic pseudo coordinates inside Bangladesh bbox.


    This keeps the map usable until backend provides real geo centroids.


    """


    if df.empty:


        return df


    lat_col = next((c for c in ["lat", "latitude", "Lat", "Latitude"] if c in df.columns), None)


    lon_col = next((c for c in ["lon", "lng", "longitude", "Longitude", "Lng"] if c in df.columns), None)


    if lat_col and lon_col:


        return df


    lat_min, lat_max = 20.5, 26.7


    lon_min, lon_max = 88.0, 92.7


    def _coord(name: str):


        h = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12], 16)


        r1 = (h % 10_000_000) / 10_000_000


        r2 = ((h // 10_000_000) % 10_000_000) / 10_000_000


        lat = lat_min + r1 * (lat_max - lat_min)


        lon = lon_min + r2 * (lon_max - lon_min)


        return lat, lon


    lats, lons = [], []


    for _, row in df.iterrows():


        lat, lon = _coord(row.get("region") or row.get("Region") or row.name)


        lats.append(lat); lons.append(lon)


    df = df.copy()


    df["lat"] = lats


    df["lon"] = lons


    return df





# Division centroids (approximate) for Dhaka, Chattogram, Rajshahi, Khulna, Sylhet,


# Barishal, Mymensingh, Rangpur. Source: common references; fine for dashboard bubbles.


DIVISION_CENTROIDS = {
    "Dhaka": (23.8103, 90.4125),
    "Chittagong": (22.3569, 91.7832),
    "Rajshahi": (24.3636, 88.6241),
    "Khulna": (22.8456, 89.5403),
    "Sylhet": (24.8949, 91.8687),
    "Barisal": (22.7010, 90.3535),
    "Mymensingh": (24.7471, 90.4203),
    "Rangpur": (25.7439, 89.2752),
}

DIVISION_ALIASES = {
    "dhaka": "Dhaka",
    "chittagong": "Chittagong",
    "chattogram": "Chittagong",
    "chattagram": "Chittagong",
    "rajshahi": "Rajshahi",
    "khulna": "Khulna",
    "sylhet": "Sylhet",
    "barishal": "Barisal",
    "barisal": "Barisal",
    "mymensingh": "Mymensingh",
    "rangpur": "Rangpur",
}





def _add_division_centroids(df: pd.DataFrame) -> pd.DataFrame:


    """Attach lat/lon using known division centroids; fallback to pseudo for unknown."""


    if df.empty or "division" not in df.columns:


        return df


    df = df.copy()


    lats, lons = [], []


    for _, row in df.iterrows():


        div = str(row.get("division") or "").strip()


        if div in DIVISION_CENTROIDS:


            lat, lon = DIVISION_CENTROIDS[div]


        else:


            lat, lon = _coord_for_name(div or "Unknown")  # pseudo fallback


        lats.append(lat)


        lons.append(lon)


    df["lat"] = lats


    df["lon"] = lons


    return df





def _coord_for_name(name: str) -> tuple[float, float]:


    """Shared deterministic pseudo coord for a string."""


    lat_min, lat_max = 20.5, 26.7


    lon_min, lon_max = 88.0, 92.7


    h = int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12], 16)


    r1 = (h % 10_000_000) / 10_000_000


    r2 = ((h // 10_000_000) % 10_000_000) / 10_000_000


    lat = lat_min + r1 * (lat_max - lat_min)


    lon = lon_min + r2 * (lon_max - lon_min)


    return lat, lon





DIVISION_KEYWORDS_FE = [


    ("Dhaka", ["dhaka"]),


    ("Chittagong", ["chattogram", "chattagram", "chittagong"]),


    ("Rajshahi", ["rajshahi"]),


    ("Khulna", ["khulna"]),


    ("Sylhet", ["sylhet"]),


    ("Barisal", ["barisal", "barishal"]),


    ("Mymensingh", ["mymensingh"]),


    ("Rangpur", ["rangpur"]),


]





def _infer_division_fe(text: str | None) -> str | None:


    if not text:


        return None


    t = str(text).lower()


    for div, keys in DIVISION_KEYWORDS_FE:


        if any(k in t for k in keys):


            return div


    return None





# -------------------------------------------------------------------


# Helpers


# -------------------------------------------------------------------





def call_json(method: str, url: str, *, timeout: int = 30, **kwargs):
    headers = kwargs.pop("headers", {}) or {}
    headers = {**auth_headers(), **headers}

    try:
        resp = requests.request(method=method.upper(), url=url, timeout=timeout, headers=headers, **kwargs)
    except Exception as e:
        return {"_error": f"request_failed: {e}"}

    if resp.status_code == 401:
        clear_auth()
        return {"_error": "unauthorized", "status": resp.status_code}

    try:
        resp.raise_for_status()
    except requests.HTTPError:
        return {"_error": "http_error", "status": resp.status_code, "body": (resp.text or "")[:2000]}

    try:
        return resp.json()
    except Exception:
        return {"_error": "json_decode_error", "status": resp.status_code, "body": (resp.text or "")[:2000]}








def render_login_panel():
    st.markdown(
        """
        <style>
        .login-side{
          background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 45%, #f8fafc 100%);
          border: 1px solid #e2e8f0;
          border-radius: 2px;
          padding: 26px;
          min-height: 360px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .login-logo{
          width: 64px;
          height: 64px;
          border-radius: 14px;
          background: #ffffff;
          border: 1px solid #cbd5f5;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }
        .login-logo img{
          max-width: 100%;
          max-height: 100%;
          object-fit: contain;
          display: block;
        }
        .login-side h3{
          margin: 6px 0 4px;
          font-family: 'Outfit', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
          font-weight: 700;
          color: #0f172a;
        }
        .login-side p{
          margin: 0;
          color: #475569;
          font-size: 14px;
        }
        .login-points{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
          margin-top: 8px;
        }
        .login-point{
          background: rgba(255, 255, 255, 0.7);
          border: 1px solid #e2e8f0;
          border-radius: 2px;
          padding: 10px 12px;
          font-size: 12px;
          color: #0f172a;
          font-weight: 600;
        }
        @media (max-width: 900px){
          .login-side{ min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_uri = _load_login_logo_data_uri()
    if logo_uri:
        logo_html = f'<img src="{logo_uri}" alt="Media Monitoring logo" />'
    else:
        logo_html = """
        <svg width="38" height="38" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="6" y="8" width="36" height="28" rx="6" fill="#dbeafe" stroke="#1e3a8a" stroke-width="2"/>
          <path d="M12 16h12M12 22h24M12 28h18" stroke="#1e3a8a" stroke-width="2" stroke-linecap="round"/>
        </svg>
        """
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Sign in")
        st.caption("Use your account to access the monitoring workspace.")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", key="login_password", type="password")
        if st.button("Sign in", key="login_button"):
            if not username or not password:
                st.warning("Enter username and password.")
                return
            res = call_json("POST", f"{API}/api/auth/login", json={"username": username, "password": password}, timeout=30)
            if res.get("_error"):
                st.error(f"Login failed: {res}")
            else:
                st.session_state["access_token"] = res.get("access_token")
                st.session_state["user"] = res.get("user")
                qp = {**get_query_params(), **auth_query_params()}
                _set_query_params(**qp)
                st.success("Signed in")
                st.rerun()
    with right:
        st.markdown(
            f"""
            <div class="login-side">
              <div class="login-logo">
                {logo_html}
              </div>
              <h3>News Monitoring</h3>
              <p>Track coverage trends, geography snapshots, and newsroom signals in one place.</p>
              <div class="login-points">
                <div class="login-point">Division heatmaps</div>
                <div class="login-point">Category trends</div>
                <div class="login-point">Party snapshots</div>
                <div class="login-point">Candidate tracking</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def ensure_logged_in() -> bool:
    if st.session_state.get("access_token") and st.session_state.get("user"):
        return True
    render_login_panel()
    return False


def excel_download(df: pd.DataFrame, label: str, filename: str, key: str):


    buf = io.BytesIO()


    with pd.ExcelWriter(buf, engine="openpyxl") as xw:


        df.to_excel(xw, index=False)


    buf.seek(0)


    st.download_button(label, buf.getvalue(), file_name=filename,


                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",


                       key=key)








def render_metrics(counts: dict, duplicates_override: int | None = None):


    labels = [
        ("Total", counts.get("total", 0)),
        ("Duplicates", counts.get("duplicates", 0) if duplicates_override is None else duplicates_override),
        ("Unclassified", counts.get("unclassified", 0)),
        ("Queued", counts.get("queued", 0)),
        ("Classified", counts.get("classified", 0)),
    ]

    cols = st.columns(len(labels))
    for col, (label, val) in zip(cols, labels):
        col.metric(label, f"{val:,}")








def render_tile_grid(items, label_key="category", count_key="primary", cols=4, label_transform=None):


    for i in range(0, len(items), cols):


        row = st.columns(cols)


        for col, item in zip(row, items[i:i+cols]):


            raw_label = item.get(label_key) or "Unknown"


            label = label_transform(raw_label) if label_transform else raw_label


            count = item.get(count_key, 0)


            total = item.get("total", count)


            col.metric(label, f"{count:,}", help=f"Total: {total:,}")



def render_candidate_cards(items, limit=10):
    """
    Show top candidates as cards with candidate/party, news counts, and a top category hint.
    """
    top = items[:limit]
    if not top:
        return
    cols = st.columns(4)
    for idx, item in enumerate(top):
        col = cols[idx % 4]
        name = item.get("candidate") or "Unknown"
        party_raw = item.get("party") or "Unknown"
        party_disp = _translate_bn(party_raw) if SHOW_BN else party_raw
        if not str(party_disp).strip():
            party_disp = "Unknown"

        # Prefer Mentions/total over primary (older payloads)
        count_raw = item.get("Mentions") or item.get("total") or item.get("primary") or 0
        try:
            news_count = int(float(count_raw))
        except Exception:
            news_count = 0

        by_cat = item.get("by_category") or {}
        top_cats = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)[:3]
        top_cat_label = display_cat(top_cats[0][0]) if top_cats else ("ক্যাটাগরি নেই" if SHOW_BN else "No category")
        cat_text = ", ".join(f"{display_cat(k)} ({v})" for k, v in top_cats) if top_cats else "—"
        card = f"""
        <div style="
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border:1px solid #e2e8f0;
            border-radius:14px;
            padding:14px 16px;
            box-shadow:0 10px 25px rgba(15,23,42,0.06);
            margin-bottom:14px;
        ">
          <div style="display:flex; gap:12px; align-items:center;">
            <div style="width:56px;height:56px;flex-shrink:0;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;background:#f1f5f9;">
              <img src="https://placehold.co/112x112?text=NA" style="width:100%;height:100%;object-fit:cover;">
            </div>
            <div style="flex:1;">
              <div style="font-weight:700;font-size:15px;color:#0f172a;">{name}</div>
              <div style="font-size:12px;color:#64748b;">Party: {party_disp}</div>
            </div>
          </div>
          <div style="display:flex;gap:10px;margin-top:10px;font-size:13px;color:#334155;">
            <span>News Count: <strong>{news_count}</strong></span>
          </div>
          <div style="margin-top:10px;font-size:12px;color:#475569;">
            Top topic: <strong>{top_cat_label}</strong><br/>
            By category: {cat_text}
          </div>
        </div>
        """
        col.markdown(card, unsafe_allow_html=True)






def fetch_counts(params: dict | None = None):
    res = call_json("GET", f"{API}/api/counts", params=params or {}, timeout=20)
    return res if isinstance(res, dict) else {}


def fetch_stats(endpoint: str, params: dict | None = None):
    res = call_json("GET", f"{API}{endpoint}", params=params or {}, timeout=60)
    return res.get("items", []) if isinstance(res, dict) else []





def fetch_stats_with_params(endpoint: str, params: dict | None = None):


    res = call_json("GET", f"{API}{endpoint}", params=params or {}, timeout=60)


    return res.get("items", []) if isinstance(res, dict) else []





def _agg_division_counts(df: pd.DataFrame) -> pd.DataFrame:


    if df.empty:


        return df


    work = df.copy()


    if "division" not in work.columns:


        return df


    work["division"] = work["division"].fillna("Unknown")


    if "primary" in work.columns:


        work["Mentions"] = pd.to_numeric(work["primary"], errors="coerce").fillna(0)


    elif "total" in work.columns:


        work["Mentions"] = pd.to_numeric(work["total"], errors="coerce").fillna(0)


    else:


        work["Mentions"] = 0


    agg = work.groupby("division", as_index=False)["Mentions"].sum()


    return agg


def _ensure_division_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'division' column exists by borrowing region or inferring."""
    if df.empty or "division" in df.columns:
        return df
    if "region" in df.columns:
        df = df.copy()
        df["division"] = df["region"].apply(lambda v: _normalize_division(_infer_division_fe(v) or v))
        return df
    return df


def _hydrate_divisions(div_df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee all divisions are present with counts (default 0) for mapping/labels."""
    base = pd.DataFrame({"division": list(DIVISION_CENTROIDS.keys()), "Mentions": 0})
    if div_df is None or div_df.empty:
        return base
    work = div_df.copy()
    if "division" not in work.columns:
        return base
    work["division"] = work["division"].apply(_normalize_division)
    work["Mentions"] = pd.to_numeric(work.get("Mentions", 0), errors="coerce").fillna(0)
    agg = work.groupby("division", as_index=False)["Mentions"].sum()
    merged = base.merge(agg, on="division", how="left", suffixes=("_base", ""))
    merged["Mentions"] = merged["Mentions"].fillna(merged["Mentions_base"]).fillna(0)
    return merged[["division", "Mentions"]]


def _prepare_geo_map_df(div_df: pd.DataFrame, label: str | None = None, extra: str | None = None) -> pd.DataFrame:
    """Attach color, size, and tooltip helpers for geo maps."""
    if div_df.empty:
        return div_df

    df = div_df.copy()
    df["Mentions"] = pd.to_numeric(df.get("Mentions", 0), errors="coerce").fillna(0)
    df["_size"] = df["Mentions"]
    max_m = max(df["_size"].max(), 1)
    light = (191, 219, 254)
    dark = (12, 74, 110)
    alpha_min, alpha_max = 90, 235

    def _color(val: float) -> list[int]:
        frac = float(val) / float(max_m) if max_m else 0
        r = int(light[0] + (dark[0] - light[0]) * frac)
        g = int(light[1] + (dark[1] - light[1]) * frac)
        b = int(light[2] + (dark[2] - light[2]) * frac)
        a = int(alpha_min + (alpha_max - alpha_min) * frac)
        return [r, g, b, a]

    label_txt = (label or "").strip()
    extra_txt = (extra or "").strip()

    def _tooltip(row) -> str:
        parts = [
            f"Region: {row.get('division', 'Unknown') or 'Unknown'}",
            f"Mentions: {int(row.get('_size', 0))}",
        ]
        if label_txt:
            parts.append(f"Category: {label_txt}")
        if extra_txt:
            parts.append(extra_txt)
        return "<br>".join(parts)

    df["_fill"] = df["_size"].apply(_color)
    df["_tooltip"] = df.apply(_tooltip, axis=1)
    scale_max = math.log(max_m + 1) if max_m else 1

    def _radius(val: float) -> int:
        if val <= 0:
            return 5000
        frac = math.log(val + 1) / scale_max if scale_max else 0
        return int(5000 + (frac * 15000))

    df["_radius"] = df["_size"].apply(_radius)
    return df


def _normalize_division(name: str | None) -> str:
    if not name:
        return "Unknown"
    key = str(name).strip().lower()
    return DIVISION_ALIASES.get(key, str(name).strip())


_GEOJSON_CACHE = {"path": None, "mtime": None, "data": None}


def _ring_area(ring: list[list[float]]) -> float:
    if len(ring) < 3:
        return 0.0
    area = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        area += (x1 * y2) - (x2 * y1)
    return abs(area / 2.0)


def _polygon_area(poly: list[list[list[float]]]) -> float:
    if not poly:
        return 0.0
    area = _ring_area(poly[0])
    for hole in poly[1:]:
        area -= _ring_area(hole)
    return abs(area)


def _ring_bbox_area(ring: list[list[float]]) -> float:
    if not ring:
        return 0.0
    min_lon = min(pt[0] for pt in ring)
    max_lon = max(pt[0] for pt in ring)
    min_lat = min(pt[1] for pt in ring)
    max_lat = max(pt[1] for pt in ring)
    return max(0.0, (max_lon - min_lon) * (max_lat - min_lat))


def _simplify_ring(ring: list[list[float]]) -> list[list[float]]:
    if not ring or len(ring) < 4:
        return ring
    # Remove consecutive duplicates.
    dedup = [ring[0]]
    for pt in ring[1:]:
        if pt != dedup[-1]:
            dedup.append(pt)
    if len(dedup) < 4:
        return dedup
    step = 1
    if len(dedup) > 2000:
        step = 4
    elif len(dedup) > 1000:
        step = 3
    elif len(dedup) > 500:
        step = 2
    if step > 1:
        dedup = dedup[::step]
    if dedup[0] != dedup[-1]:
        dedup.append(dedup[0])
    return dedup


def _clean_geojson(gj: dict) -> dict:
    cleaned = dict(gj)
    cleaned_features = []
    for feat in gj.get("features", []):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords or gtype not in ("Polygon", "MultiPolygon"):
            continue
        polys = [coords] if gtype == "Polygon" else coords
        areas = [_polygon_area(p) for p in polys]
        max_area = max(areas) if areas else 0.0
        keep_min = max_area * 0.01 if max_area else 0.0
        new_polys = []
        for poly, area in zip(polys, areas):
            if area < keep_min:
                continue
            new_poly = []
            for ring in poly:
                simp = _simplify_ring(ring)
                if simp and len(simp) >= 4:
                    ring_area = _ring_area(simp)
                    bbox_area = _ring_bbox_area(simp)
                    if bbox_area > 0 and ring_area / bbox_area < 0.005:
                        continue
                    new_poly.append(simp)
            if new_poly:
                new_polys.append(new_poly)
        if not new_polys:
            continue
        new_geom = {"type": gtype, "coordinates": new_polys[0] if gtype == "Polygon" else new_polys}
        new_feat = dict(feat)
        new_feat["geometry"] = new_geom
        cleaned_features.append(new_feat)
    cleaned["features"] = cleaned_features
    return cleaned


def _load_division_geojson():
    """
    Load Bangladesh division GeoJSON; prefer curated BD_Divisions.json (GitHub source),
    fall back to legacy bangladesh_divisions.geojson if present.
    """
    candidates = ["BD_Divisions.json", "bangladesh_divisions.geojson"]
    data_dirs = [
        Path(__file__).resolve().parent.parent / "data",
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path("/data"),
    ]
    for data_dir in data_dirs:
        for name in candidates:
            path = data_dir / name
            if not path.exists():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            cache = _GEOJSON_CACHE
            if cache["path"] == str(path) and cache["mtime"] == stat.st_mtime and cache["data"] is not None:
                return cache["data"]
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                data = _clean_geojson(data)
                _GEOJSON_CACHE.update({"path": str(path), "mtime": stat.st_mtime, "data": data})
                return data
            except Exception:
                continue
    return None


def _build_geojson_choropleth(div_df: pd.DataFrame, label: str | None = None, extra: str | None = None):
    """Merge division counts into the real division GeoJSON (Bangladesh admin-1)."""
    gj = _load_division_geojson()
    if gj is None:
        return None

    df = div_df.copy()
    if "division" not in df.columns:
        df["division"] = None
    df["division_norm"] = df["division"].apply(_normalize_division)
    df["Mentions"] = pd.to_numeric(df.get("Mentions", 0), errors="coerce").fillna(0)
    count_map = df.groupby("division_norm")["Mentions"].sum().to_dict()
    max_m = max(count_map.values()) if count_map else 0
    max_m = max(max_m, 1)

    light = (219, 234, 254)  # light blue for empty/low
    dark = (30, 64, 175)     # deep blue for high
    alpha_min, alpha_max = 120, 235
    scale_max = max_m if max_m else 1

    def _color(val: float) -> list[int]:
        if val <= 0:
            return [light[0], light[1], light[2], alpha_min]
        frac = min(float(val) / float(scale_max), 1.0) if scale_max else 0.0
        r = int(light[0] + (dark[0] - light[0]) * frac)
        g = int(light[1] + (dark[1] - light[1]) * frac)
        b = int(light[2] + (dark[2] - light[2]) * frac)
        a = int(alpha_min + (alpha_max - alpha_min) * frac)
        return [r, g, b, a]

    label_txt = (label or "").strip()
    extra_txt = (extra or "").strip()

    gj_copy = json.loads(json.dumps(gj))
    for feat in gj_copy.get("features", []):
        props = feat.get("properties", {})
        name = props.get("ADM1_EN") or props.get("shapeName") or props.get("division") or "Unknown"
        key = _normalize_division(name)
        mentions = int(count_map.get(key, 0))
        tooltip_lines = [
            f"Division: {name}",
            f"Mentions: {mentions}",
        ]
        if label_txt:
            tooltip_lines.append(f"Category: {label_txt}")
        if extra_txt:
            tooltip_lines.append(extra_txt)
        props["Mentions"] = mentions
        props["fill"] = _color(mentions)
        tooltip_html = "<br>".join(tooltip_lines)
        props["tooltip"] = tooltip_html
        props["division"] = name
        feat["properties"] = props
        # Duplicate key fields at top-level for tooltip compatibility.
        feat["division"] = name
        feat["Mentions"] = mentions
        feat["tooltip"] = tooltip_html
    return gj_copy


def _geo_map_key(prefix: str, df: pd.DataFrame, start_dt: date, end_dt: date, extra: str | None = None) -> str:
    """Stable-but-changing key so pydeck rerenders when counts change."""
    sig = "none"
    if not df.empty and "division" in df.columns and "Mentions" in df.columns:
        rows = df[["division", "Mentions"]].copy()
        rows["Mentions"] = pd.to_numeric(rows["Mentions"], errors="coerce").fillna(0).astype(int)
        pairs = [f"{d}:{m}" for d, m in rows.sort_values("division").itertuples(index=False, name=None)]
        sig = hashlib.sha1("|".join(pairs).encode("utf-8")).hexdigest()[:10] if pairs else "empty"
    extra_txt = (extra or "").strip().replace(" ", "_")
    return f"{prefix}_{start_dt}_{end_dt}_{extra_txt}_{sig}"


def fetch_articles(params: dict):


    res = call_json("GET", f"{API}/api/articles/search", params=params, timeout=120)


    return res.get("items", []) if isinstance(res, dict) else []


SEARCH_LIMIT_MAX = 2000
SEARCH_OFFSET_MAX = 5000


def fetch_articles_batched(limit: int | None, params: dict | None = None, chunk_size: int = 800):
    """
    Fetch articles in batches to honor API limit/offset caps (limit<=2000, offset<=5000).
    limit=0 means fetch until we exhaust results or hit offset cap.
    """
    limit = int(limit or 0)
    params = params or {}
    chunk_size = max(1, min(SEARCH_LIMIT_MAX, int(chunk_size)))
    items = []
    offset = 0
    truncated = False
    error = None

    while True:
        if limit > 0 and len(items) >= limit:
            break
        remaining = (limit - len(items)) if limit > 0 else chunk_size
        batch_limit = max(1, min(chunk_size, remaining if limit > 0 else chunk_size, SEARCH_LIMIT_MAX))
        req_params = {**params, "limit": batch_limit, "offset": offset}
        res = call_json("GET", f"{API}/api/articles/search", params=req_params, timeout=120)
        if res.get("_error"):
            error = res
            break
        batch_items = res.get("items", []) if isinstance(res, dict) else []
        items.extend(batch_items)
        if len(batch_items) < batch_limit:
            break
        offset += batch_limit
        if offset >= SEARCH_OFFSET_MAX:
            truncated = True
            break

    return {"items": items, "truncated": truncated, "error": error}


def fetch_classified_articles_all(params: dict | None = None, batch_size: int = 800):
    """
    Fetch all classified articles (primary labels) using the dedicated endpoint.
    Honors the API caps (limit<=2000, offset<=5000) and reports truncation.
    """
    params = params or {}
    items = []
    offset = 0
    truncated = False
    error = None
    batch_size = max(1, min(2000, int(batch_size)))

    while True:
        req_params = {**params, "limit": batch_size, "offset": offset}
        res = call_json("GET", f"{API}/api/articles/classified", params=req_params, timeout=120)
        if res.get("_error"):
            error = res
            break
        batch_items = res.get("items", []) if isinstance(res, dict) else []
        items.extend(batch_items)
        if len(batch_items) < batch_size:
            break
        offset += batch_size
        if offset >= 5000:  # API offset cap
            truncated = True
            break

    return {"items": items, "truncated": truncated, "error": error}


def _extract_headline_text(item: dict) -> str:
    """Pick the best available headline/title field for deduplication."""
    if not isinstance(item, dict):
        return ""
    for key in ("title", "Headline", "headline", "header", "news_title", "Title", "name"):
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _extract_link(item: dict) -> str:
    """Best-effort URL for an article record."""
    if not isinstance(item, dict):
        return ""
    for key in ("url", "link", "source_url", "href", "news_url"):
        val = item.get(key)
        if val:
            return str(val)
    return ""


def _normalize_headline(text: str) -> str:
    """Normalize headline text so small punctuation/casing differences collapse."""
    if not text:
        return ""
    txt = str(text)
    txt = txt.replace("“", "\"").replace("”", "\"").replace("’", "'")
    txt = re.sub(r"https?://\S+", " ", txt)
    txt = re.sub(r"[^\w\s]", " ", txt.lower())
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _token_signature(normalized: str) -> str:
    tokens = sorted(set(normalized.split()))
    return " ".join(tokens)


def _published_dt(row: dict):
    if not isinstance(row, dict):
        return pd.NaT
    for key in ("published_at", "published", "date", "Date"):
        val = row.get(key)
        if val is None:
            continue
        try:
            dt = pd.to_datetime(val, errors="coerce", utc=True)
        except Exception:
            dt = pd.NaT
        if pd.isna(dt):
            continue
        return dt
    return pd.NaT


def _row_identifier(row: dict) -> str:
    if not isinstance(row, dict):
        return hashlib.sha256(str(row).encode("utf-8")).hexdigest()
    for key in ("article_id", "id", "_id", "url", "link"):
        val = row.get(key)
        if val:
            return str(val)
    try:
        return hashlib.sha256(str(sorted(row.items())).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(repr(row).encode("utf-8")).hexdigest()


def _choose_canonical(items):
    fallback_ts = pd.Timestamp(1970, 1, 1, tz="UTC")
    def sort_key(item):
        ts = _published_dt(item.get("row") or {})
        if pd.isna(ts):
            ts = fallback_ts
        return (ts, len(item.get("title") or ""))
    return max(items, key=sort_key)


def dedupe_articles_by_headline(items, enable_fuzzy: bool = True, fuzzy_threshold: float = 0.9):
    """
    Deduplicate articles by normalized headline; returns summary with duplicate groups.
    """
    if not items:
        return {"total": 0, "scanned": 0, "skipped": 0, "duplicates": 0, "unique": 0, "groups": []}

    enriched = []
    skipped = 0
    total = len(items) if isinstance(items, list) else 0

    for rec in items:
        if not isinstance(rec, dict):
            continue
        title_raw = _extract_headline_text(rec)
        norm = _normalize_headline(title_raw)
        if not norm:
            skipped += 1
            continue
        enriched.append(
            {
                "row": rec,
                "title": title_raw,
                "norm": norm,
                "token": _token_signature(norm),
            }
        )

    groups = {}
    dup_ids = set()

    def ensure_group(base_item):
        cid = _row_identifier(base_item.get("row") or {})
        if cid not in groups:
            groups[cid] = {"canonical": base_item, "duplicates": []}
        return groups[cid]

    # Exact-match deduplication
    exact_index = defaultdict(list)
    for item in enriched:
        exact_index[item["norm"]].append(item)

    for _, bucket in exact_index.items():
        if len(bucket) < 2:
            continue
        base = _choose_canonical(bucket)
        base_id = _row_identifier(base.get("row") or {})
        group = ensure_group(base)
        for dup in bucket:
            dup_id = _row_identifier(dup.get("row") or {})
            if dup_id == base_id or dup_id in dup_ids:
                continue
            dup_ids.add(dup_id)
            group["duplicates"].append({"item": dup, "reason": "exact", "score": 1.0})

    # Fuzzy/token deduplication
    if enable_fuzzy:
        token_index = defaultdict(list)
        for item in enriched:
            token_index[item["token"]].append(item)
        for _, bucket in token_index.items():
            if len(bucket) < 2:
                continue
            base = _choose_canonical(bucket)
            base_id = _row_identifier(base.get("row") or {})
            group = ensure_group(base)
            for dup in bucket:
                dup_id = _row_identifier(dup.get("row") or {})
                if dup_id == base_id or dup_id in dup_ids:
                    continue
                score = SequenceMatcher(None, base["norm"], dup["norm"]).ratio()
                if score < fuzzy_threshold:
                    continue
                dup_ids.add(dup_id)
                group["duplicates"].append({"item": dup, "reason": "fuzzy", "score": score})

    groups_list = [g for g in groups.values() if g["duplicates"]]
    groups_list.sort(key=lambda g: len(g["duplicates"]), reverse=True)

    duplicates = len(dup_ids)
    scanned = len(enriched)
    unique = max(scanned - duplicates, 0)

    return {
        "total": total,
        "scanned": scanned,
        "skipped": skipped,
        "duplicates": duplicates,
        "unique": unique,
        "groups": groups_list,
    }






def fetch_candidates(limit: int = 500):


    res = call_json("GET", f"{API}/api/candidates", params={"limit": limit}, timeout=30)


    return res.get("items", []) if isinstance(res, dict) else []





def fetch_candidate_refs():


    res = call_json("GET", f"{API}/api/candidate_refs", timeout=30)


    return res.get("items", []) if isinstance(res, dict) else []





def auto_tag_candidates(limit: int, min_hits: int, seeds=None, mode: str = "heuristic"):
    limit = int(limit)
    # LLM/hybrid runs can be slow; cap batch size and extend timeout to avoid client read timeouts.
    if mode == "llm":
        limit = min(limit, 60)
    elif mode == "hybrid":
        limit = min(limit, 200)
    timeout = 600 if mode in ("llm", "hybrid") else 240

    return call_json(
        "POST",
        f"{API}/api/auto/tag_candidates",
        params={"limit": limit, "min_hits": int(min_hits), "mode": mode},
        json={"seeds": seeds or []},
        timeout=timeout,
    )





# -------------------------------------------------------------------


# Category translation (English  Bangla) for display


# -------------------------------------------------------------------





@lru_cache(maxsize=256)


def _translate_bn(text: str) -> str:


    """


    Lightweight translation using the free Google endpoint.


    Cached to avoid repeated network calls.


    """


    if not text:


        return ""


    t = str(text).strip()


    if not t:


        return ""


    # If already contains Bangla characters, return as-is


    if any("\u0980" <= ch <= "\u09ff" for ch in t):


        return t


    try:


        resp = requests.get(


            "https://translate.googleapis.com/translate_a/single",


            params={"client": "gtx", "sl": "auto", "tl": "bn", "dt": "t", "q": t},


            timeout=8,


        )


        resp.raise_for_status()


        data = resp.json()


        translated = "".join(seg[0] for seg in data[0] if seg[0])


        return translated or t


    except Exception:


        return t





SHOW_BN = st.sidebar.checkbox("Show categories in Bangla", True, help="Translate category labels to Bangla for display.")





def display_cat(label: str | None) -> str:


    if not label:


        return "Unknown"


    return _translate_bn(label) if SHOW_BN else label


_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_BN_MONTHS = [
    "জানুয়ারি",
    "ফেব্রুয়ারি",
    "মার্চ",
    "এপ্রিল",
    "মে",
    "জুন",
    "জুলাই",
    "আগস্ট",
    "সেপ্টেম্বর",
    "অক্টোবর",
    "নভেম্বর",
    "ডিসেম্বর",
]

def format_date_bn(val):
    """
    Render a date/time value with Bangla digits; returns empty string if missing.
    """
    if val is None:
        return ""
    try:
        from datetime import datetime, date
        if isinstance(val, datetime):
            txt = val.strftime("%Y-%m-%d")
        elif isinstance(val, date):
            txt = val.strftime("%Y-%m-%d")
        else:
            txt = str(val)
        return txt.translate(_BN_DIGITS)
    except Exception:
        return str(val)


def format_date_pretty(val, use_bn: bool = False):
    """
    Human-friendly date like '০৫ ডিসেম্বর ২০২৫, ২১:৪৮' (Bangla) or '05 Dec 2025, 21:48'.
    Assumes UTC input if naive; converts to Asia/Dhaka when possible.
    """
    if val is None:
        return ""
    try:
        dt = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.isna(dt):
            return ""
        # Convert to Dhaka; if already tz-aware pandas handles it
        try:
            dt = dt.tz_convert("Asia/Dhaka")
        except Exception:
            pass
        day = dt.day
        month = dt.month
        year = dt.year
        hour = dt.hour
        minute = dt.minute
        if use_bn:
            day_txt = f"{day:02d}".translate(_BN_DIGITS)
            year_txt = f"{year}".translate(_BN_DIGITS)
            month_txt = _BN_MONTHS[month - 1] if 1 <= month <= 12 else ""
            time_txt = f"{hour:02d}:{minute:02d}".translate(_BN_DIGITS)
            return f"{day_txt} {month_txt} {year_txt}, {time_txt}"
        month_en = dt.strftime("%b")
        return f"{day:02d} {month_en} {year}, {hour:02d}:{minute:02d}"
    except Exception:
        return str(val)







# -------------------------------------------------------------------


# Pages


# -------------------------------------------------------------------





def page_dashboard():


    st.title("Dashboard")


    counts = fetch_counts()

    empty_dedupe = {"total": 0, "scanned": 0, "skipped": 0, "duplicates": 0, "unique": 0, "groups": []}

    # Headline-based duplicate count across all news (batched; 0 = all until offset cap).
    dup_all_summary = {"duplicates": counts.get("duplicates", 0)}
    with st.spinner("Counting duplicates across all news..."):
        batch_all = fetch_articles_batched(limit=0, params={}, chunk_size=800)
        if batch_all.get("error"):
            st.error(f"Failed to fetch articles for duplicate count: {batch_all['error']}")
            dup_all_items = []
        else:
            dup_all_items = batch_all.get("items", [])
            if batch_all.get("truncated"):
                st.warning("Stopped early due to API offset cap (5000). Duplicate count may be underreported.")
        dup_all_summary = dedupe_articles_by_headline(
            dup_all_items,
            enable_fuzzy=True,
            fuzzy_threshold=0.9,
        ) if dup_all_items else {**empty_dedupe, "total": 0}

    render_metrics(counts, duplicates_override=dup_all_summary.get("duplicates", 0))

    st.subheader("Duplicate news (headline-based)")
    st.caption("Counts duplicates using normalized headline/title fields (exact and optional fuzzy match).")
    _, _, dup_start_dt, dup_end_dt = timeline_control("dash_dup_tl")
    dup_date_params = _date_params(dup_start_dt, dup_end_dt)
    dup_ctrl_cols = st.columns([1.1, 1.0, 1.0])
    max_articles = max(counts.get("total", 0) or 0, 1000)
    default_articles = min(300, max_articles) if max_articles else 0
    step_articles = max(10, max_articles // 20) if max_articles else 10
    with dup_ctrl_cols[0]:
        dup_limit = st.slider(
            "Articles to scan",
            0,
            int(max_articles),
            int(default_articles),
            int(step_articles),
            key="dash_dup_limit",
            help="0 = scan all available in window (may be slow)",
        )
    with dup_ctrl_cols[1]:
        dup_fuzzy = st.checkbox("Enable fuzzy match (near duplicate)", True, key="dash_dup_fuzzy")
    with dup_ctrl_cols[2]:
        dup_threshold = st.slider(
            "Fuzzy threshold",
            75,
            100,
            90,
            1,
            key="dash_dup_threshold",
            help="Similarity percent (token/sequence) required to mark as duplicate.",
        )

    dup_summary = dict(empty_dedupe)
    dup_items = []
    with st.spinner("Scanning headlines for duplicates..."):
        fetch_res = fetch_articles_batched(
            limit=int(dup_limit),
            params=dup_date_params,
            chunk_size=800,
        )
        if fetch_res.get("error"):
            st.error(f"Failed to fetch articles for dedupe: {fetch_res['error']}")
        else:
            dup_items = fetch_res.get("items", [])
            if fetch_res.get("truncated"):
                st.warning("Stopped early due to API offset cap (5000). Duplicate count may be underreported.")
            dup_summary = dedupe_articles_by_headline(
                dup_items,
                enable_fuzzy=dup_fuzzy,
                fuzzy_threshold=max(0.0, min(1.0, dup_threshold / 100.0)),
            ) if dup_items else dict(empty_dedupe)

    total_news_all = counts.get("total", 0)
    total_dups_all = dup_all_summary.get("duplicates", counts.get("duplicates", 0))
    overall_cols = st.columns(2)
    overall_cols[0].metric("Total news (all)", f"{total_news_all:,}")
    overall_cols[1].metric("Total duplicates (all)", f"{total_dups_all:,}")

    dup_metrics = st.columns(5)
    dup_metrics[0].metric("Fetched", f"{dup_summary.get('total', len(dup_items)):,}")
    dup_metrics[1].metric("Scanned (with headline)", f"{dup_summary.get('scanned', 0):,}")
    dup_metrics[2].metric("Skipped (no headline)", f"{dup_summary.get('skipped', 0):,}")
    dup_metrics[3].metric("Likely duplicates", f"{dup_summary.get('duplicates', 0):,}")
    dup_metrics[4].metric("Unique headlines", f"{dup_summary.get('unique', 0):,}")

    dup_groups = dup_summary.get("groups") or []
    if dup_groups:
        rows = []
        for grp in dup_groups[:20]:
            base_item = grp.get("canonical") or {}
            base_row = base_item.get("row") or {}
            base_title = base_item.get("title") or _extract_headline_text(base_row)
            base_portal = base_row.get("portal") or base_row.get("source") or base_row.get("Newspaper") or ""
            base_link = _extract_link(base_row)
            base_pub = format_date_pretty(base_row.get("published_at") or base_row.get("published"), use_bn=SHOW_BN)
            for dup in grp.get("duplicates", []):
                dup_row = dup.get("item", {}).get("row") or {}
                dup_title = dup.get("item", {}).get("title") or _extract_headline_text(dup_row)
                dup_link = _extract_link(dup_row)
                match_txt = "exact" if dup.get("reason") == "exact" else f"fuzzy {int(round(dup.get('score', 0) * 100))}%"
                rows.append(
                    {
                        "Canonical headline": base_title,
                        "Duplicate headline": dup_title,
                        "Canonical link": base_link,
                        "Duplicate link": dup_link,
                        "Match": match_txt,
                        "Canonical source": base_portal,
                        "Duplicate source": dup_row.get("portal") or dup_row.get("source") or dup_row.get("Newspaper") or "",
                        "Published": format_date_pretty(
                            dup_row.get("published_at") or dup_row.get("published"), use_bn=SHOW_BN
                        ) or base_pub,
                    }
                )
        if rows:
            df_dup = pd.DataFrame(rows)
            st.dataframe(df_dup.head(100), use_container_width=True, hide_index=True)
            if len(rows) > 100:
                st.caption(f"Showing first 100 of {len(rows)} duplicate rows.")
        else:
            st.info("No duplicates found in the scanned window.")
    else:
        st.info("No duplicates found in the scanned window.")

    st.subheader("Upload XLSX")
    up_c1, up_c2, up_c3 = st.columns([1.6, 1.1, 0.9])
    with up_c1:
        upload_file = st.file_uploader("News XLSX (e.g., News.xlsx)", type=["xlsx", "xls"], key="dash_upload_xlsx")
    with up_c2:
        upload_portal = st.text_input("Default portal (fallback when Newspaper missing)", value="Uploaded XLSX", key="dash_upload_portal")
    with up_c3:
        upload_lang = st.text_input("Portal language", value="bn", key="dash_upload_lang")
    st.caption("Columns supported: Newspaper, Date, Headline, Content, Link, Elections (Headline/Content required).")
    if st.button("Upload and ingest", key="dash_upload_btn"):
        if not upload_file:
            st.warning("Choose a file to upload.")
        else:
            try:
                payload = upload_file.getvalue()
            except Exception:
                payload = upload_file.read()
            files = {
                "file": (
                    upload_file.name,
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            }
            data = {"portal_name": upload_portal, "portal_lang": upload_lang}
            with st.spinner("Uploading and ingesting..."):
                res = call_json("POST", f"{API}/api/admin/upload_xlsx", files=files, data=data, timeout=900)
            if res.get("_error"):
                st.error(res)
            else:
                st.success(f"Inserted {res.get('inserted',0)} of {res.get('rows',0)} rows; skipped={res.get('skipped',0)}")
                if res.get("warnings"):
                    with st.expander("Warnings", expanded=False):
                        for w in res.get("warnings", [])[:200]:
                            st.write(w)


    st.subheader("Bulk actions")


    col_a, col_b, col_c = st.columns([1.4, 1.4, 1.0])





    with col_a:


        st.caption("Classify unclassified (queued/new/needs_review)")


        cls_mode_options = {


            "Auto (rule+ml+llm)": "auto",


            "Hybrid (rule+ml)": "hybrid",


            "Rule only": "rule",


            "ML only": "ml",


            "LLM fallback": "llm",


        }


        cls_mode_label = st.selectbox("Mode", list(cls_mode_options.keys()), index=0, key="cls_mode")


        cls_batch = st.slider("Batch size", 10, 500, 200, 10, key="cls_batch")


        if st.button("Classify News", key="btn_classify"):


            with st.spinner("Classifying news..."):
                res = call_json(
                    "POST",
                    f"{API}/api/batch/classify",
                    params={"limit": int(cls_batch), "mode": cls_mode_options[cls_mode_label]},
                    timeout=900,
                )

            if "_error" in res:


                st.error(res)


            else:


                processed = int(res.get("processed", 0) or 0)
                picked = int(res.get("picked", 0) or 0)
                failed = int(res.get("failed", 0) or 0)
                categorized = int(res.get("categorized", processed - failed) or 0)

                st.success(f"Processed {processed} of {picked}; categorized={categorized}; failed={failed}")

                classified_items = res.get("classified_items") or []
                if classified_items:
                    st.caption(f"Classified articles (showing {len(classified_items)} rows)")
                    df_cls = pd.DataFrame(classified_items)

                    # Prefer per-mode terms if present, otherwise fall back to category keywords
                    kw_source_col = "terms" if "terms" in df_cls.columns else ("mode_terms" if "mode_terms" in df_cls.columns else ("keywords" if "keywords" in df_cls.columns else None))
                    if kw_source_col:
                        df_cls["terms"] = df_cls[kw_source_col].apply(
                            lambda ks: ", ".join(ks[:10]) if isinstance(ks, list) else (ks or "")
                        )
                    if "published_at" in df_cls:
                        df_cls["published_at"] = df_cls["published_at"].apply(
                            lambda v: format_date_pretty(v, use_bn=SHOW_BN)
                        )
                    if "score" in df_cls:
                        df_cls["score"] = df_cls["score"].apply(
                            lambda v: round(float(v), 3) if pd.notnull(v) else None
                        )

                    display_cols = [
                        "article_id",
                        "title",
                        "url",
                        "category",
                        "terms",
                        "mode",
                        "model_used",
                        "portal",
                        "published_at",
                        "party",
                        "candidate",
                        "region",
                        "score",
                    ]
                    display_cols = [c for c in display_cols if c in df_cls.columns]
                    rename_map = {
                        "article_id": "Article ID",
                        "title": "Title",
                        "url": "URL",
                        "category": "Category",
                        "terms": "Terms",
                        "mode": "Mode",
                        "model_used": "Model/Source",
                        "portal": "Portal",
                        "published_at": "Published",
                        "party": "Party",
                        "candidate": "Candidate",
                        "region": "Region",
                        "score": "Score",
                    }
                    display_df = df_cls[display_cols].rename(columns=rename_map)
                    display_df.insert(0, "Run At", datetime.now())

                    prev_df = st.session_state.get("cls_results_df")
                    if isinstance(prev_df, pd.DataFrame) and not prev_df.empty:
                        combined = pd.concat([display_df, prev_df], ignore_index=True)
                    else:
                        combined = display_df
                    combined = combined.drop_duplicates(subset=["Article ID"], keep="first")
                    if "Published" in combined.columns:
                        combined = combined.sort_values(by=["Run At", "Published"], ascending=[False, False], na_position="last")
                    else:
                        combined = combined.sort_values(by=["Run At"], ascending=False)
                    st.session_state["cls_results_df"] = combined.reset_index(drop=True)
                    st.dataframe(
                        st.session_state["cls_results_df"],
                        use_container_width=True,
                        hide_index=True,
                        height=560,
                        column_config={
                            "URL": st.column_config.LinkColumn("URL"),
                            "Run At": st.column_config.DatetimeColumn("Run At"),
                            "Published": st.column_config.TextColumn("Published"),
                            "Keywords/Terms": st.column_config.TextColumn("Keywords/Terms"),
                        },
                    )
                else:
                    st.session_state["cls_results_df"] = pd.DataFrame()
                    st.caption("No classified article details returned for this run.")





    with col_b:


        st.caption("Auto-generate categories via clustering (creates auto_category rows)")


        gen_limit = st.slider("Articles to cluster (0 = all)", 0, 1000, 200, 50, key="gen_limit")


        gen_mcs = st.slider("Min cluster size", 5, 100, 20, 1, key="gen_mcs")


        if st.button("Auto Generate Category", key="btn_generate"):


            params = {"embeddings": True, "min_cluster_size": int(gen_mcs)}


            if gen_limit > 0:


                params["limit"] = int(gen_limit)


            res = call_json("POST", f"{API}/api/auto/init", params=params, timeout=3600)


            if "_error" in res:


                st.error(res)


                with st.expander("Details", expanded=False):


                    st.write(res.get("body") or res)


            else:


                st.success("Auto categories generation triggered.")


                with st.expander("Logs (tail)", expanded=False):


                    st.text_area("Logs", value=(res.get("logs") or "")[-3000:], height=220)





    with col_c:
        st.caption("Deduplicate by URL hash")
        if st.button("Mark duplicates", key="btn_dedupe"):
            with st.spinner("Checking duplicates..."):
                res = call_json("POST", f"{API}/api/admin/dedupe/urlhash", timeout=120)
            if res.get("_error"):
                st.error(res)
            else:
                st.success(f"Marked {res.get('updated',0)} duplicate articles across {res.get('groups',0)} groups.")

    cls_df_latest = st.session_state.get("cls_results_df")
    if isinstance(cls_df_latest, pd.DataFrame) and not cls_df_latest.empty:
        st.markdown("#### Classified articles (recent)")
        st.dataframe(
            cls_df_latest,
            use_container_width=True,
            hide_index=True,
            height=560,
            column_config={
                "URL": st.column_config.LinkColumn("URL"),
                "Run At": st.column_config.DatetimeColumn("Run At"),
                "Published": st.column_config.TextColumn("Published"),
                "Keywords/Terms": st.column_config.TextColumn("Keywords/Terms"),
            },
        )

    st.subheader("Auto-tag candidates")


    candidate_refs = fetch_candidate_refs()


    col_c1, col_c2, col_c3 = st.columns([1.2, 1.2, 1.2])


    with col_c1:


        cand_limit = st.slider("Max untagged articles to scan", 10, 500, 100, 10, key="cand_auto_limit")


    with col_c2:


        min_hits = st.slider("Min keyword hits (heuristic)", 1, 5, 1, 1, key="cand_auto_hits")


    with col_c3:


        mode_label = st.selectbox("Method", ["Heuristic (aliases)", "LLM extraction", "Hybrid (heuristic  LLM)"], key="cand_auto_mode")


        if mode_label.startswith("Heuristic"):


            mode_val = "heuristic"


        elif mode_label.startswith("LLM"):


            mode_val = "llm"


        else:


            mode_val = "hybrid"


    seed_input = st.text_input("Seed candidates (comma separated, optional)", key="cand_auto_seeds",


                               help="Required if candidate list is empty.")


    if not candidate_refs and not seed_input and mode_val == "heuristic":


        st.warning("Add candidates to the list below or provide seeds to auto-tag.")


    if st.button("Auto Tag Candidates", key="btn_auto_tag_cand"):


        seeds = [s.strip() for s in seed_input.split(",") if s.strip()] if seed_input else []


        res = auto_tag_candidates(cand_limit, min_hits, seeds=seeds, mode=mode_val)


        if "_error" in res:


            st.error(res)


        else:


            msg = f"Tagged {res.get('tagged',0)} of {res.get('scanned',0)} scanned."


            if not res.get("ok", True):


                msg += f" (reason: {res.get('reason')})"


            st.success(msg)


            if res.get("items"):


                st.caption("Sample (max 50):")


                st.dataframe(pd.DataFrame(res["items"]), use_container_width=True, hide_index=True)





    st.subheader("Auto-generated categories")


    auto = call_json("GET", f"{API}/api/auto/categories", timeout=60)


    if isinstance(auto, dict) and not auto.get("_error"):


        auto_items = auto.get("items", [])


        if auto_items:


            st.caption("Rationale shows cluster size, representative terms, and which model/algorithm chose the label; promote to classify all linked news.")


            df_auto = pd.DataFrame([{


                "id": it.get("id"),


                "label": it.get("label"),


                "size": it.get("size"),


                "top_terms": ", ".join(it.get("top_terms") or []),


                "algo": it.get("algo"),


                "model": it.get("model_name"),


                "why_label": (


                    lambda terms, algo, model, size: (


                        f"Based on {size or 'n/a'} articles; representative terms: "


                        f"{'; '.join([f'{i+1}) {t}' for i, t in enumerate(terms[:8])]) or 'n/a'}; "


                        f"algorithm: {algo or 'n/a'}; embedding/model: {model or 'n/a'}"


                    )


                )(


                    list(it.get("top_terms") or []),


                    it.get("algo"),


                    it.get("model_name"),


                    it.get("size"),


                ),


            } for it in auto_items])


            if SHOW_BN:


                df_auto.insert(2, "label_bn", df_auto["label"].apply(display_cat))


            st.dataframe(df_auto, use_container_width=True, hide_index=True)


            excel_download(df_auto, "Download Auto Categories", "auto_categories.xlsx", key="auto_download")





            opts = [f"{it.get('id')}: {display_cat(it.get('label'))} (size {it.get('size',0)})" for it in auto_items]


            sel_auto = st.selectbox("Promote an auto category to main categories (classify linked news)", opts)


            if st.button("Promote selected auto category", key="promote_auto_dash"):


                try:


                    sel_id = int(sel_auto.split(":", 1)[0])


                    res = call_json("POST", f"{API}/api/auto/promote/{sel_id}", timeout=180)


                    if "_error" in res:


                        st.error(res)


                    else:


                        st.success(f"Promoted as '{res.get('promoted_label')}', assigned {res.get('assigned',0)} articles")


                except Exception as e:


                    st.error(f"Promote failed: {e}")


        else:


            st.info("No auto-generated categories yet. Run Auto Generate Category from Dashboard.")


    else:


        st.warning(f"Could not load auto categories: {auto}")





    st.subheader("Top Categories")


    cats = fetch_stats("/api/stats/categories", params={"primary_only": True})


    if cats:


        render_tile_grid(cats[:8], label_key="category", label_transform=display_cat)


        df = pd.DataFrame(cats)


        if SHOW_BN and not df.empty and "category" in df:


            df.insert(1, "category_bn", df["category"].apply(display_cat))


        st.dataframe(df, use_container_width=True, hide_index=True)


    else:


        st.info("No category stats yet.")





    st.subheader("Top Parties")


    parties = fetch_stats("/api/stats/party", params={"primary_only": True})


    if parties:


        render_tile_grid(parties[:8], label_key="party", count_key="primary")


    else:


        st.info("No party stats yet.")





    st.subheader("Top Candidates")


    candidates = fetch_stats("/api/stats/candidate", params={"primary_only": True})


    if candidates:


        render_candidate_cards(candidates, limit=10)


    else:


        st.info("No candidate stats yet.")





    st.subheader("Recent Articles")


    items = fetch_articles({"limit": 20})


    if items:


        df = pd.DataFrame(items)[["article_id", "title", "published_at", "status", "category", "portal"]]


        if SHOW_BN:


            df.insert(4, "category_bn", df["category"].apply(display_cat))
            df["published_at"] = df["published_at"].apply(format_date_bn)
        else:
            df["published_at"] = df["published_at"].astype(str)


        st.dataframe(df, use_container_width=True, hide_index=True)


    else:


        st.info("No articles found.")








    st.subheader("Categories")
    cats_res = call_json("GET", f"{API}/api/categories", timeout=30)
    cats_raw = cats_res.get("items") if isinstance(cats_res, dict) else []

    rows = []
    for item in cats_raw or []:
        if isinstance(item, str):
            rows.append({
                "Category": item,
                "Top terms": "",
                "Model": "",
                "Algorithm": "",
                "Auto": "No",
            })
            continue
        cat = (item or {}).get("category") or ""
        terms = item.get("top_terms") or []
        if isinstance(terms, str):
            terms = [terms]
        top_terms_txt = ", ".join([str(t).strip() for t in terms if str(t).strip()])
        auto_flag = "Yes" if (item.get("is_auto") or item.get("auto")) else "No"
        rows.append({
            "Category": cat,
            "Top terms": top_terms_txt,
            "Model": item.get("model_name") or "",
            "Algorithm": item.get("algo") or "",
            "Auto": auto_flag,
        })

    if rows:
        df_cats = pd.DataFrame(rows)
        if SHOW_BN:
            df_cats.insert(1, "Category (bn)", df_cats["Category"].apply(display_cat))
        st.dataframe(df_cats, use_container_width=True, hide_index=True)
        excel_download(df_cats, "Download categories", "categories.xlsx", key="dash_download_categories")
    else:
        st.info("No categories found.")



    st.subheader("Classified Articles")
    cls_fetch = fetch_classified_articles_all()
    cls_items = cls_fetch.get("items") or []
    if cls_items:
        df_cls_dash = pd.DataFrame(cls_items)
        kw_src = "terms" if "terms" in df_cls_dash.columns else ("mode_terms" if "mode_terms" in df_cls_dash.columns else ("keywords" if "keywords" in df_cls_dash.columns else None))
        if kw_src:
            df_cls_dash["Terms"] = df_cls_dash[kw_src].apply(
                lambda ks: ", ".join(ks[:12]) if isinstance(ks, list) else (ks or "")
            )
        if "published_at" in df_cls_dash:
            df_cls_dash["Published"] = df_cls_dash["published_at"].apply(
                lambda v: format_date_pretty(v, use_bn=SHOW_BN)
            )
        if "category" in df_cls_dash:
            df_cls_dash["category_bn"] = df_cls_dash["category"].apply(display_cat)
        display_cols = [
            "article_id",
            "title",
            "url",
            "Published",
            "portal",
            "category",
            "category_bn",
            "Terms",
            "mode",
            "party",
            "candidate",
            "region",
        ]
        display_cols = [c for c in display_cols if c in df_cls_dash.columns]
        rename_map = {
            "article_id": "Article ID",
            "title": "Title",
            "url": "URL",
            "portal": "Portal",
            "category": "Category",
            "category_bn": "Category (bn)",
            "Terms": "Terms",
            "mode": "Mode",
            "party": "Party",
            "candidate": "Candidate",
            "region": "Region",
        }
        df_cls_dash = df_cls_dash[display_cols].rename(columns=rename_map)
        if cls_fetch.get("truncated"):
            st.caption("Showing first 5,000 classified articles due to API offset limits.")
        st.caption(f"Total classified shown: {len(df_cls_dash)}")
        st.dataframe(
            df_cls_dash,
            use_container_width=True,
            hide_index=True,
            column_config={
                "URL": st.column_config.LinkColumn("URL"),
                "Published": st.column_config.TextColumn("Published"),
                "Keywords/Terms": st.column_config.TextColumn("Keywords/Terms"),
            },
        )
    else:
        st.info("No classified articles available.")


def _render_party_like(title: str, stats_endpoint: str, key_field: str):


    st.title(title)


    stats = fetch_stats(stats_endpoint)


    if not stats:


        st.info("No data yet.")


        return





    options = ["All"] + [s[key_field] for s in stats if s.get(key_field)]


    selected = st.selectbox(f"Select {key_field}", options)


    st.subheader("Summary")


    df_stats = pd.DataFrame(stats)


    st.dataframe(df_stats, use_container_width=True, hide_index=True)





    st.subheader("News")


    params = {"limit": 200}


    if selected != "All":


        params[key_field] = selected


    items = fetch_articles(params)


    if items:


        df = pd.DataFrame(items)[["article_id", "title", "published_at", "category", "portal", "party", "candidate", "region"]]


        if SHOW_BN:


            df.insert(3, "category_bn", df["category"].apply(display_cat))


        st.dataframe(df, use_container_width=True, hide_index=True)


        excel_download(df, "Download Excel", f"{key_field}_news.xlsx", key=f"{key_field}_download")


    else:


        st.info("No articles match the filters.")








def page_party():


    st.markdown("### Party")


    _, _, start_dt, end_dt = timeline_control("party_tl")
    date_params = _date_params(start_dt, end_dt)





    if "party_selected" not in st.session_state:


        st.session_state["party_selected"] = "All"





    stats = fetch_stats("/api/stats/party", params={**date_params, "primary_only": True})





    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Party News Counts (click to filter)")

    if "party_show_all" not in st.session_state:
        st.session_state["party_show_all"] = False

    toggle_cols = st.columns([5, 1.2])
    with toggle_cols[1]:
        if st.button("Show all parties" if not st.session_state["party_show_all"] else "Show top 20", key="party_toggle_btn"):
            st.session_state["party_show_all"] = not st.session_state["party_show_all"]
            st.rerun()

    stats_sorted = sorted(stats or [], key=lambda s: float(s.get("primary", 0) or 0), reverse=True)
    stats_visible = stats_sorted if st.session_state["party_show_all"] else stats_sorted[:20]

    cards = [{"party": "All", "primary": sum(s.get("primary", 0) for s in stats_sorted)}] + stats_visible


    for i in range(0, len(cards), 4):


        cols = st.columns(4)


        for col, card in zip(cols, cards[i : i + 4]):


            label = card.get("party") or "Unknown"


            val = _format_big_number(card.get("primary", 0))


            is_active = st.session_state.get("party_selected", "All") == label


            btn_label = f"{label}  {val}"


            with col:


                if st.button(btn_label, key=f"party_card_{i}_{label}", type="primary" if is_active else "secondary", use_container_width=True):


                    st.session_state["party_selected"] = label


                    st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)





    selected_party = st.session_state.get("party_selected", "All")




    st.markdown('<div class="section">', unsafe_allow_html=True)


    heading_party = "All parties" if selected_party == "All" else selected_party


    st.markdown(f"#### Category Counts ({heading_party})")


    row_cols = st.columns(2)


    cat_counts = {}


    if stats:


        source_items = stats if selected_party == "All" else [


            s for s in stats if (s.get("party") or "Unknown") == selected_party


        ]


        for item in source_items:


            for cat, cnt in (item.get("by_category") or {}).items():


                cat_key = cat or "Unknown"


                try:


                    cnt_val = float(cnt)


                except Exception:


                    cnt_val = 0


                cat_counts[cat_key] = cat_counts.get(cat_key, 0) + cnt_val


    cat_df = pd.DataFrame(


        [{"Category": display_cat(cat), "Count": cnt} for cat, cnt in cat_counts.items()]


    )


    if not cat_df.empty:

        cat_df["Count"] = pd.to_numeric(cat_df["Count"], errors="coerce").fillna(0).astype(int)

        cat_df = cat_df.sort_values("Count", ascending=False)

        with row_cols[0]:
            heat_rows = []
            for s in source_items:
                party_name = s.get("party") or "Unknown"
                for cat, cnt in (s.get("by_category") or {}).items():
                    label = display_cat(cat) if SHOW_BN else (cat or "Unknown")
                    try:
                        cnt_val = float(cnt)
                    except Exception:
                        cnt_val = 0
                    heat_rows.append({"Party": party_name, "Category": label, "Count": cnt_val})

            df_heat = pd.DataFrame(heat_rows)
            if not df_heat.empty:
                # Aggregate in case of duplicates, then keep top categories to avoid oversized charts
                df_heat = df_heat.groupby(["Party", "Category"], as_index=False)["Count"].sum()
                top_cats = (
                    df_heat.groupby("Category")["Count"].sum().sort_values(ascending=False).head(12).index.tolist()
                )
                df_heat = df_heat[df_heat["Category"].isin(top_cats)]
            if not df_heat.empty:
                base_chart = alt.Chart(df_heat).encode(
                    x=alt.X("Category:N", sort="-y", title="Category"),
                    y=alt.Y("Party:N", title="Party"),
                )
                heatmap = base_chart.mark_rect(cornerRadius=2).encode(
                    color=alt.Color(
                        "Count:Q",
                        scale=alt.Scale(scheme="viridis", type="linear"),
                        title="Mentions",
                    ),
                    tooltip=["Party:N", "Category:N", "Count:Q"],
                )
                chart = heatmap.properties(height=400).configure_axis(
                    labelFontSize=11, titleFontSize=12
                ).configure_view(
                    strokeWidth=0
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No category data available to render heatmap.")

        with row_cols[1]:
            st.dataframe(cat_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    else:
        st.info("No category data available for the selected party.")


    st.markdown('</div>', unsafe_allow_html=True)





    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Party News Feed")


    # Temporarily ignore date range for feed (use full dataset)
    params = {"limit": 200, **date_params}


    if selected_party != "All":


        params["party"] = selected_party


    items = fetch_articles(params)


    if items:


        records = []


        for rec in items:


            rec = dict(rec)


            if SHOW_BN:


                rec["category"] = display_cat(rec.get("category"))


            records.append(rec)


        cards_html = "\n".join(


            _news_card_html(


                {


                    "title": r.get("title"),


                    "category": r.get("category"),


                    "party": r.get("party"),


                    "published_at": r.get("published_at"),


                    "url": r.get("url") or "#",


                    "portal": r.get("portal"),


                    "description": "",


                },


                show_image=False,


            )


            for r in records


        )


        grid_html = '<div class="news-card-grid">\n' + cards_html + "\n</div>"
        scrollable = f'<div style="max-height:640px; overflow-y:auto; padding-right:8px;">{grid_html}</div>'

        st.markdown(scrollable, unsafe_allow_html=True)


        excel_download(pd.DataFrame(items), "Download Excel", "party_news.xlsx", key="party_download")


    else:


        st.info("No articles match the selected party.")


    st.markdown('</div>', unsafe_allow_html=True)








def page_candidate():


    st.markdown("### Candidate")


    _, _, start_dt, end_dt = timeline_control("cand_tl")
    date_params = _date_params(start_dt, end_dt)





    stats = fetch_stats("/api/stats/candidate", params={**date_params, "primary_only": True})


    candidate_refs = fetch_candidate_refs()


    party_stats = fetch_stats("/api/stats/party", params={**date_params, "primary_only": True})





    party_options = ["All"] + [p["party"] for p in party_stats if p.get("party")]


    base_constituencies = {
        (r.get("seat") or "").strip()
        for r in candidate_refs
        if (r.get("seat") or "").strip()
    }
    base_constituencies = {c for c in base_constituencies if c}
    constituency_options = ["All"] + sorted(base_constituencies) if base_constituencies else ["All"]


    party_sel = "All"


    constituency_sel = "All"





    # Build mapping candidate -> party/seat from refs


    ref_party = {r.get("name") or r.get("name_bn"): r.get("party") for r in candidate_refs}


    ref_seat = {r.get("name") or r.get("name_bn"): r.get("seat") for r in candidate_refs}





    df_stats = pd.DataFrame(stats) if stats else pd.DataFrame(columns=["candidate", "primary"])


    if not df_stats.empty:


        df_stats = df_stats.rename(columns={"primary": "Mentions"})


        df_stats["Mentions"] = pd.to_numeric(df_stats["Mentions"], errors="coerce").fillna(0)


        df_stats["party"] = df_stats["candidate"].map(ref_party)
        df_stats["party"] = df_stats["party"].fillna("Unknown")


        df_stats["seat"] = df_stats["candidate"].map(ref_seat)


    else:


        df_stats = pd.DataFrame(columns=["candidate", "Mentions", "party", "seat"])





    # Top 10 candidates cards


    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Top 10 Candidates")

    top_cands = df_stats.sort_values("Mentions", ascending=False).head(10)
    if not top_cands.empty:
        render_candidate_cards(top_cands.to_dict(orient="records"), limit=10)
    else:
        st.info("No candidates found.")


    st.markdown('</div>', unsafe_allow_html=True)





    # Candidate news volume (bar chart + table split)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("#### Candidate News Volume")
    chart_cols = st.columns([1, 1])
    if not df_stats.empty:
        chart_df = df_stats.sort_values("Mentions", ascending=False)
        with chart_cols[0]:
            st.bar_chart(chart_df.head(20).set_index("candidate")["Mentions"])
        with chart_cols[1]:
            st.dataframe(
                chart_df[["candidate", "Mentions", "party", "seat"]].head(20),
                use_container_width=True,
                hide_index=True,
            )
    else:
        with chart_cols[0]:
            st.info("No candidate data available.")
    st.markdown('</div>', unsafe_allow_html=True)





    # News filters and feed


    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Candidate News Feed")


    cand_options = ["All"] + sorted([c for c in df_stats["candidate"].dropna().unique().tolist()]) if not df_stats.empty else ["All"]
    news_filters = st.columns([1, 1, 1, 1, 1])

    with news_filters[0]:
        nf_cand = st.selectbox("Candidate", cand_options)

    with news_filters[1]:
        nf_party = st.selectbox("Party", party_options, index=party_options.index(party_sel) if party_sel in party_options else 0)

    feed_params = {"limit": 200, **date_params}
    if nf_party != "All":
        feed_params["party"] = nf_party
    if nf_cand != "All":
        feed_params["candidate"] = nf_cand

    feed_items = fetch_articles(feed_params)

    region_constituencies = sorted({
        str(r.get("region") or "").strip()
        for r in feed_items
        if str(r.get("region") or "").strip()
    })
    if region_constituencies:
        merged = set(constituency_options[1:])
        merged.update(region_constituencies)
        constituency_options = ["All"] + sorted(merged)

    with news_filters[2]:
        nf_const = st.selectbox(
            "Constituency",
            constituency_options,
            index=constituency_options.index(constituency_sel) if constituency_sel in constituency_options else 0,
        )

    with news_filters[3]:
        nf_div = st.selectbox("Division", ["All"] + list(DIVISION_CENTROIDS.keys()))

    with news_filters[4]:
        nf_dist = st.text_input("District contains", "")

    # client-side filters


    filtered_feed = []


    for r in feed_items:


        div_val = _infer_division_fe(r.get("region"))


        if nf_div != "All" and div_val != nf_div:


            continue


        if nf_dist and nf_dist.lower() not in str(r.get("region") or "").lower():


            continue


        cand_name = r.get("candidate")
        cand_seat = str(ref_seat.get(cand_name) or "").strip()
        region_val = str(r.get("region") or "").strip()
        if nf_cand != "All" and cand_name != nf_cand:
            continue
        if nf_const != "All":
            target = str(nf_const).strip().lower()
            if target not in {cand_seat.lower(), region_val.lower()}:
                continue

        rec = dict(r)
        if SHOW_BN and rec.get("category"):
            rec["category"] = display_cat(rec.get("category"))
        filtered_feed.append(rec)





    if filtered_feed:

        cards_html = "\n".join(_news_card_html(rec, show_image=False) for rec in filtered_feed)

        grid_html = '<div class="news-card-grid">\n' + cards_html + "\n</div>"
        scrollable = f'<div style="max-height:640px; overflow-y:auto; padding-right:8px;">{grid_html}</div>'

        st.markdown(scrollable, unsafe_allow_html=True)

    else:


        st.info("No news found for the selected filters.")


    st.markdown('</div>', unsafe_allow_html=True)








def page_geography():


    st.title("Geography")

    _, _, start_dt, end_dt = timeline_control("geo_tl")
    date_params = _date_params(start_dt, end_dt)

    stats = fetch_stats("/api/stats/geography", params={**date_params, "primary_only": True})

    geo_df = pd.DataFrame(stats) if stats else pd.DataFrame()
    geo_df = _ensure_division_column(geo_df)
    div_options = ["All"] + sorted(
        [d for d in geo_df.get("division", []).dropna().unique().tolist()] if "division" in geo_df else []
    )
    selected = st.selectbox("Select region", div_options)
    if selected != "All" and not geo_df.empty:
        geo_df = geo_df[
            geo_df["division"].fillna("").str.lower() == str(selected).lower()
        ]


    st.markdown('<div class="section party-snapshot">', unsafe_allow_html=True)


    st.markdown("#### Geography Snapshot")


    geo_cols = st.columns(2)


    div_df = _agg_division_counts(geo_df)
    div_df = _hydrate_divisions(div_df)
    div_df = _add_division_centroids(div_df)
    map_extra = f"Region filter: {selected}" if selected != "All" else ""
    div_df = _prepare_geo_map_df(div_df, label="All categories", extra=map_extra)
    geojson = _build_geojson_choropleth(div_df, label="All categories", extra=map_extra)


    with geo_cols[0]:


        st.caption("Map view")


        lat_col = next((c for c in ["lat", "latitude", "Lat", "Latitude"] if c in div_df.columns), None)


        lon_col = next((c for c in ["lon", "lng", "longitude", "Longitude", "Lng"] if c in div_df.columns), None)


        view_lat = 23.685
        view_lon = 90.3563
        if lat_col and lon_col and not div_df.empty:
            top = div_df.loc[div_df["Mentions"].idxmax()]
            view_lat = float(top[lat_col])
            view_lon = float(top[lon_col])
            if selected != "All":
                pick = div_df[div_df["division"].str.lower() == str(selected).lower()]
                if not pick.empty:
                    view_lat = float(pick.iloc[0][lat_col])
                    view_lon = float(pick.iloc[0][lon_col])
        layers = []
        if geojson:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson,
                    stroked=False,
                    filled=True,
                    extruded=False,
                    wireframe=False,
                    opacity=0.9,
                    get_fill_color="properties.fill",
                    get_line_color=[0, 0, 0, 0],
                    lineWidthScale=0,
                    lineWidthMinPixels=0,
                    pickable=True,
                    auto_highlight=True,
                    highlightColor=[255, 230, 120, 120],
                )
            )
        elif lat_col and lon_col and not div_df.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=div_df,
                    get_position=[lon_col, lat_col],
                    get_radius="_radius",
                    radius_min_pixels=3,
                    radius_max_pixels=18,
                    get_fill_color="_fill",
                    pickable=True,
                )
            )

        if layers:


            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=5, pitch=0),
                layers=layers,
                tooltip={
                    "html": "{tooltip}" if geojson else "{_tooltip}",
                    "style": {
                        "fontSize": "12px",
                        "color": "#f8fafc",
                        "backgroundColor": "#0f172a",
                        "padding": "8px 10px",
                    },
                },
            )
            map_key = _geo_map_key("geo_map", div_df, start_dt.date(), end_dt.date(), extra=selected)
            st.pydeck_chart(deck, key=map_key)


        else:


            st.info("No geo coordinates available to render map.")


    with geo_cols[1]:


        st.caption("Location-wise news count")


        if not div_df.empty:


            df_geo_table = div_df[["division", "Mentions"]].copy().sort_values("Mentions", ascending=False)


            st.dataframe(df_geo_table, use_container_width=True, hide_index=True)


        else:


            st.info("No geography data available.")


    st.markdown('</div>', unsafe_allow_html=True)


    params = {"limit": 200, **date_params}


    if selected != "All":


        params["region"] = selected


    items = fetch_articles(params)


    st.subheader("News list")


    if items:


        df_news = pd.DataFrame(items)[["article_id", "title", "published_at", "category", "portal", "region", "party", "candidate"]]


        if SHOW_BN:


            df_news.insert(3, "category_bn", df_news["category"].apply(display_cat))


        st.dataframe(df_news, use_container_width=True, hide_index=True)


        excel_download(df_news, "Download Excel", "geography_news.xlsx", key="geo_download")


    else:


        st.info("No articles match the filters.")








def page_media():


    st.title("Media")

    _, _, start_dt, end_dt = timeline_control("media_tl")
    date_params = _date_params(start_dt, end_dt)

    stats = fetch_stats("/api/stats/media", params={**date_params, "primary_only": True})


    portal_names = ["All"] + [s["portal"] for s in stats]


    selected = st.selectbox("Select media", portal_names)





    if stats:


        df = pd.DataFrame(stats)


        st.dataframe(df, use_container_width=True, hide_index=True)


    else:


        st.info("No media stats yet.")





    params = {"limit": 200, **date_params}


    if selected != "All":


        match = next((s for s in stats if s["portal"] == selected), None)


        if match:


            params["portal_id"] = match["portal_id"]


    items = fetch_articles(params)


    st.subheader("News list")


    if items:


        df_news = pd.DataFrame(items)[["article_id", "title", "published_at", "category", "portal", "party", "candidate", "region"]]


        if SHOW_BN:


            df_news.insert(3, "category_bn", df_news["category"].apply(display_cat))


        st.dataframe(df_news, use_container_width=True, hide_index=True)


        excel_download(df_news, "Download Excel", "media_news.xlsx", key="media_download")


    else:


        st.info("No articles match the filters.")








# Rebuilt Category page layout (aligned with media_monitoring_app)


def page_category():


    st.markdown("### Overview")


    _, _, start_dt, end_dt = timeline_control("cat_tl")
    date_params = _date_params(start_dt, end_dt)





    # Track selected category in session for cross-section sync


    if "cat_selected" not in st.session_state:


        st.session_state["cat_selected"] = "All"





    counts = fetch_counts(date_params)
    total_news = counts.get("total", 0) or 0
    likely_duplicates = 0
    if total_news:
        dup_cache = st.session_state.setdefault("cat_likely_dup_cache", {})
        cache_key = f"{date_params.get('start_date', '')}_{date_params.get('end_date', '')}_fuzzy90"
        cached = dup_cache.get(cache_key)
        if cached is not None:
            likely_duplicates = cached
        else:
            fetch_res = fetch_articles_batched(limit=0, params=date_params, chunk_size=800)
            if not fetch_res.get("error"):
                dup_summary = dedupe_articles_by_headline(
                    fetch_res.get("items", []),
                    enable_fuzzy=True,
                    fuzzy_threshold=0.9,
                )
                likely_duplicates = dup_summary.get("duplicates", 0) or 0
            dup_cache[cache_key] = likely_duplicates
    try:
        deduped_news = max(int(total_news) - int(likely_duplicates), 0)
    except Exception:
        deduped_news = 0

    kpi_cards = [
        ("Total News", total_news, "Mentions in this window"),
        ("Total Unique News", counts.get("unclassified", 0), "Unique items in this window"),
        ("Total News From District", counts.get("queued", 0), "District-sourced items"),
        ("News Deduplication", deduped_news, "Total news minus headline-based likely duplicates."),
    ]


    card_cols = st.columns(len(kpi_cards))


    for col, (label, value, hint) in zip(card_cols, kpi_cards):


        with col:


            render_html_block(
                f"""
                <div class="kpi">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{_format_big_number(value)}</div>
                    <div class="kpi-hint">{hint}</div>
                </div>
                """
            )





    cats = fetch_stats("/api/stats/categories", params={**date_params, "primary_only": True})


    cat_names = ["All"] + [c["category"] for c in cats] if cats else ["All"]





    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Category Snapshot (click to filter)")


    cards = [{"category": "All", "primary": counts.get("total", 0)}] + (cats or [])


    if cards:


        for i in range(0, len(cards), 4):


            cols = st.columns(4)


            for col, card in zip(cols, cards[i : i + 4]):


                label_raw = card.get("category") or "Unknown"


                label_disp = "All categories" if label_raw == "All" else display_cat(label_raw)


                val = _format_big_number(card.get("primary", 0))


                is_active = st.session_state.get("cat_selected", "All") == label_raw


                btn_label = f"{label_disp}  {val}"


                with col:
                    if st.button(btn_label, key=f"cat_card_{i}_{label_raw}", type="primary" if is_active else "secondary", use_container_width=True):


                        st.session_state["cat_selected"] = label_raw


                        st.rerun()


    else:


        st.info("No category stats yet.")


    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section cat-snapshot">', unsafe_allow_html=True)

    st.markdown("#### Category Counts")

    if cats:

        df_cat = pd.DataFrame(cats)
        selected_cat = st.session_state.get("cat_selected", "All")

        if not df_cat.empty and "primary" in df_cat:

            df_cat = df_cat[["category", "primary"]].rename(columns={"primary": "Mentions"})
            if selected_cat != "All":
                df_cat = df_cat[df_cat["category"].str.lower() == str(selected_cat).lower()]

            df_cat["Mentions"] = pd.to_numeric(df_cat["Mentions"], errors="coerce").fillna(0)
            df_cat = df_cat[df_cat["Mentions"] > 0].sort_values("Mentions", ascending=False)

            if SHOW_BN:
                df_cat["label"] = df_cat["category"].apply(display_cat)
                df_cat = df_cat.set_index("label")
            else:
                df_cat = df_cat.set_index("category")

            if not df_cat.empty:
                st.bar_chart(df_cat, y="Mentions")
            else:
                st.info("No category data available for this filter.")

        else:

            st.info("No category data available.")

    else:

        st.info("No category data available.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section party-snapshot">', unsafe_allow_html=True)

    st.markdown("#### Geography Snapshot")


    geo_cols = st.columns(2)


    selected_cat = st.session_state.get("cat_selected", "All")


    geo_params = {"primary_only": True}


    if selected_cat != "All":


        geo_params["category"] = selected_cat


    geo_params.update(date_params)
    geo_stats = fetch_stats_with_params("/api/stats/geography", geo_params)


    geo_df = pd.DataFrame(geo_stats) if geo_stats else pd.DataFrame()


    # division-level aggregation for map/table


    geo_df = _ensure_division_column(geo_df)


    div_df = _agg_division_counts(geo_df)
    div_df = _hydrate_divisions(div_df)


    # add centroids per division; fallback to pseudo


    div_df = _add_division_centroids(div_df)


    map_label = "All categories" if selected_cat == "All" else (display_cat(selected_cat) if SHOW_BN else selected_cat)
    div_df = _prepare_geo_map_df(div_df, label=map_label)
    geojson = _build_geojson_choropleth(div_df, label=map_label)


    with geo_cols[0]:


        st.caption("Map view")


        lat_col = next((c for c in ["lat", "latitude", "Lat", "Latitude"] if c in div_df.columns), None)
        lon_col = next((c for c in ["lon", "lng", "longitude", "Longitude", "Lng"] if c in div_df.columns), None)

        view_lat = 23.685
        view_lon = 90.3563
        if lat_col and lon_col and not div_df.empty:
            top = div_df.loc[div_df["Mentions"].idxmax()]
            view_lat = float(top[lat_col])
            view_lon = float(top[lon_col])

        layers: list[pdk.Layer] = []
        if geojson:
            layers.append(
                pdk.Layer(
                    "GeoJsonLayer",
                    data=geojson,
                    stroked=False,
                    filled=True,
                    extruded=False,
                    wireframe=False,
                    opacity=0.9,
                    get_fill_color="properties.fill",
                    get_line_color=[0, 0, 0, 0],
                    lineWidthScale=0,
                    lineWidthMinPixels=0,
                    pickable=True,
                    auto_highlight=True,
                    highlightColor=[255, 230, 120, 120],
                )
            )
        elif lat_col and lon_col and not div_df.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=div_df,
                    get_position=[lon_col, lat_col],
                    get_radius="_radius",
                    radius_min_pixels=3,
                    radius_max_pixels=18,
                    get_fill_color="_fill",
                    pickable=True,
                )
            )

        if layers:


            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=5, pitch=0),
                layers=layers,
                tooltip={
                    "html": "{tooltip}" if geojson else "{_tooltip}",
                    "style": {
                        "fontSize": "12px",
                        "color": "#f8fafc",
                        "backgroundColor": "#0f172a",
                        "padding": "8px 10px",
                    },
                },
            )

            map_key = _geo_map_key("cat_geo_map", div_df, start_dt.date(), end_dt.date(), extra=map_label)
            st.pydeck_chart(deck, key=map_key)


        else:


            st.info("No geo coordinates available to render map.")


    with geo_cols[1]:


        st.caption("Location-wise news count")


        if not div_df.empty:


            df_geo_table = div_df[["division", "Mentions"]].copy().sort_values("Mentions", ascending=False)


            st.dataframe(df_geo_table, use_container_width=True, hide_index=True)


        else:


            st.info("No geography data available.")


    st.markdown('</div>', unsafe_allow_html=True)





    st.markdown('<div class="section">', unsafe_allow_html=True)


    st.markdown("#### Top News")


    c1, c2 = st.columns([1.2, 2.2])


    with c1:


        dropdown_val = st.selectbox(


            "Filter by category",


            cat_names,


            index=cat_names.index(st.session_state.get("cat_selected", "All")) if st.session_state.get("cat_selected", "All") in cat_names else 0,


            format_func=lambda c: ("All categories" if c == "All" else display_cat(c)),


        )


        if dropdown_val != st.session_state.get("cat_selected", "All"):


            st.session_state["cat_selected"] = dropdown_val


            st.rerun()


    with c2:


        title_filter = st.text_input("Filter by headline")





    params = {"limit": 40, **date_params}


    if st.session_state.get("cat_selected", "All") != "All":


        params["category"] = st.session_state["cat_selected"]


    if title_filter.strip():


        params["title_contains"] = title_filter.strip()





    items = fetch_articles(params)


    if items:


        records = []


        for rec in items:


            rec = dict(rec)


            if SHOW_BN:


                rec["category"] = display_cat(rec.get("category") or rec.get("Category"))


            records.append(rec)


        max_cards = 20


        needs_scroll = len(records) > max_cards


        cards_html = "\n".join(_news_card_html(rec, show_image=False) for rec in records)


        grid_html = '<div class="news-card-grid">\n' + cards_html + "\n</div>"


        if needs_scroll:


            grid_html = f'<div class="top-news-scroll">{grid_html}</div>'


        st.markdown(grid_html, unsafe_allow_html=True)


    else:


        st.info("No articles match the filters.")


    st.markdown('</div>', unsafe_allow_html=True)







def page_admin():


    user = st.session_state.get("user") or {}
    if (user.get("role") or "").lower() != "admin":
        st.error("Admin role required to manage users.")
        return

    st.markdown("### User management")


    st.markdown("#### Add user")
    with st.form("create_user"):


        new_username = st.text_input("Username", key="new_username")


        new_display = st.text_input("Display name", key="new_display")


        new_role = st.selectbox("Role", ["user", "admin"], index=0, key="new_role")


        new_password = st.text_input("Password", type="password", key="new_password")


        new_active = st.checkbox("Active", value=True, key="new_active")


        submitted = st.form_submit_button("Create user")


        if submitted:


            if not new_username or not new_password:


                st.warning("Username and password are required.")


            else:


                res = call_json(


                    "POST",


                    f"{API}/api/admin/users",


                    json={


                        "username": new_username.strip(),


                        "display_name": new_display.strip() or None,


                        "role": new_role,


                        "password": new_password,


                        "is_active": new_active,


                    },


                    timeout=30,


                )


                if res.get("_error"):


                    st.error(res)


                else:


                    st.success("User created.")


                    st.rerun()


    users_res = call_json("GET", f"{API}/api/admin/users", timeout=30)


    if users_res.get("_error"):


        st.error(users_res)


        return


    users = users_res.get("items") or []


    st.markdown("### Existing users")


    if users:


        st.dataframe(pd.DataFrame(users)[["user_id", "username", "role", "is_active", "display_name", "last_login"]], use_container_width=True, hide_index=True)


        options = [f"{u.get('username')} (#{u.get('user_id')})" for u in users]


        sel = st.selectbox("Select user", options, index=0, key="edit_user_sel")


        selected = next((u for u in users if f"{u.get('username')} (#{u.get('user_id')})" == sel), None)


        if selected:


            edit_display = st.text_input("Display name", value=selected.get("display_name") or "", key="edit_display")


            edit_role = st.selectbox("Role", ["user", "admin"], index=(0 if (selected.get("role") or "user") == "user" else 1), key="edit_role")


            edit_active = st.checkbox("Active", value=bool(selected.get("is_active", True)), key="edit_active")


            edit_password = st.text_input("New password (optional)", type="password", key="edit_password")


            if st.button("Update user", key="update_user_btn"):


                payload = {}


                if (edit_display or "") != (selected.get("display_name") or ""):


                    payload["display_name"] = edit_display.strip()


                if (edit_role or "").lower() != (selected.get("role") or "").lower():


                    payload["role"] = edit_role


                if bool(edit_active) != bool(selected.get("is_active", True)):


                    payload["is_active"] = bool(edit_active)


                if edit_password:


                    payload["password"] = edit_password


                if not payload:


                    st.info("No changes to save.")


                else:


                    res = call_json("PATCH", f"{API}/api/admin/users/{selected.get('user_id')}", json=payload, timeout=30)


                    if res.get("_error"):


                        st.error(res)


                    else:


                        st.success("User updated.")


                        st.rerun()


    else:


        st.info("No users found.")


# -------------------------------------------------------------------


# App


# -------------------------------------------------------------------


qp = get_query_params()
if "logout" in qp:
    clear_auth(clear_all=True)
    st.rerun()


if not ensure_logged_in():
    st.stop()


current_user = st.session_state.get("user") or {}
is_admin = (current_user.get("role") or "").lower() == "admin"


PAGES = {
    "Category": page_category,
    "Party": page_party,
    "Candidate": page_candidate,
    "Geography": page_geography,
    "Media": page_media,
}

if is_admin:
    PAGES["Dashboard"] = page_dashboard
    PAGES["UserManage"] = page_admin
    PAGES["Admin"] = page_admin  # backward-compatible alias


active_page = get_page()

if active_page in ("Admin", "UserManage") and not is_admin:
    active_page = "Category"
if active_page == "Dashboard" and not is_admin:
    active_page = "Category"

render_topbar(
    active_page,
    is_admin=is_admin,
    username=current_user.get("username"),
    role=current_user.get("role"),
)

if active_page == "Dashboard":
    page_dashboard()
else:
    PAGES.get(active_page, page_category)()



st.markdown(f'<div class="footer">(c) ' + str(datetime.now().year) + ' Media Monitoring - Built with Streamlit</div>', unsafe_allow_html=True)

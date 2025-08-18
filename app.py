# =============================
# File: app.py
# Real Estate Intel — Market Radar & Lead Gen (Streamlit)
#
# Features (works out of the box in demo mode; enrich with API keys via secrets):
# - Lead Discovery: paste/upload leads; optional "AI filters" (OpenAI) for profiles
# - New Listings / Price Drops (demo CSV or connect your own source)
# - School, Zoning & Infrastructure monitors (ArcGIS + Eventbrite when tokens provided)
# - Local Events from Eventbrite (optional token) & RSS feeds (no token)
# - Mortgage Watch (FRED MORTGAGE30US; requires FRED_API_KEY or demo mode)
# - AI Daily Priorities & Content Studio (OpenAI optional)
# - Geofence Check (browser location vs hot listings; push integrations via worker.py)
# - Alerts to Slack / Email / Pushover via worker.py (GitHub Actions cron)
#
# Deployment: add requirements.txt, optional .streamlit/secrets.toml, and (optional) .github/workflows/alerts.yml
# =============================

import os
import json
import time
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
import streamlit as st

# Optional extras
try:
    import feedparser  # RSS
except Exception:
    feedparser = None

try:
    from dateutil import tz
except Exception:
    tz = None

try:
    import openai
except Exception:
    openai = None

try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None

st.set_page_config(page_title="Real Estate Intel — Market Radar", page_icon="📡", layout="wide")

# ---------------------------
# Secrets / Settings helpers
# ---------------------------

def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets.get(key)
    return os.getenv(key, default)

FRED_API_KEY = get_secret("FRED_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
EVENTBRITE_TOKEN = get_secret("EVENTBRITE_TOKEN")
SLACK_WEBHOOK_URL = get_secret("SLACK_WEBHOOK_URL")
PUSHOVER_TOKEN = get_secret("PUSHOVER_TOKEN")
PUSHOVER_USER = get_secret("PUSHOVER_USER")
SENDGRID_API_KEY = get_secret("SENDGRID_API_KEY")

if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

# ---------------------------
# Utilities
# ---------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def haversine_m(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c


def arcgis_query(service_url: str, layer: int, params: dict):
    """Generic ArcGIS FeatureServer/MapServer query with safe POST for large payloads."""
    base = f"{service_url}/{layer}/query"
    defaults = {"f":"json","where":"1=1","outFields":"*","returnGeometry":False}
    q = {**defaults, **(params or {})}
    if q.get("returnGeometry"):
        q.setdefault("outSR", 4326)
    try:
        use_post = ("geometry" in q) or (len(base) + len(str(q)) > 1800)
        if use_post:
            r = requests.post(base, data=q, timeout=30, headers={"Content-Type":"application/x-www-form-urlencoded"})
        else:
            r = requests.get(base, params=q, timeout=25)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(data.get("error"))
        return data
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# Demo / placeholder data
# ---------------------------

DEMO_LISTINGS = pd.DataFrame([
    {"id":"L-1001","address":"123 Main St, Doral, FL","price":650000,"beds":3,"baths":2,"lat":25.816,"lon":-80.355,"status":"new","url":"https://example.com/123main"},
    {"id":"L-1002","address":"456 Lake Ave, Miami, FL","price":489000,"beds":2,"baths":2,"lat":25.774,"lon":-80.19,"status":"price_drop","url":"https://example.com/456lake"},
    {"id":"L-1003","address":"789 Grove Rd, Hialeah, FL","price":399000,"beds":3,"baths":1,"lat":25.86,"lon":-80.29,"status":"new","url":"https://example.com/789grove"},
])

DEMO_LEADS = pd.DataFrame([
    {"name":"Alex R.","email":"alex@example.com","source":"Instagram","note":"Renter, turning 30 soon, wants 2/2 in Doral"},
    {"name":"Maria P.","email":"maria@example.com","source":"Zillow","note":"Saved 3 townhomes; asks about schools"},
    {"name":"Omar K.","email":"omar@example.com","source":"Open House","note":"Investor; 4-plex; cash buyer"},
])

# ---------------------------
# Connectors
# ---------------------------

def fred_mortgage_rate_series(series_id: str = "MORTGAGE30US", months_back: int = 6) -> pd.DataFrame:
    """Fetch mortgage rates from FRED. Requires FRED_API_KEY; returns demo data if missing."""
    if not FRED_API_KEY:
        # Demo mode: fabricate a small downsloping series
        end = datetime.utcnow().date()
        dates = pd.date_range(end - pd.DateOffset(months=months_back), end, freq="W-FRI")
        vals = pd.Series(7.0 - (pd.Series(range(len(dates))).astype(float) * 0.02))
        return pd.DataFrame({"date": dates, "value": vals}).reset_index(drop=True)
    url = "https://api.stlouisfed.org/fred/series/observations"
    start = (datetime.utcnow().date() - pd.DateOffset(months=months_back)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        rows = []
        for o in obs:
            v = o.get("value")
            try:
                val = float(v)
            except Exception:
                continue
            rows.append({"date": pd.to_datetime(o.get("date")), "value": val})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def eventbrite_events(query: str, location: str = "Miami, FL", days_ahead: int = 21) -> pd.DataFrame:
    """Eventbrite events (requires EVENTBRITE_TOKEN). Returns empty if not configured."""
    if not EVENTBRITE_TOKEN:
        return pd.DataFrame()
    url = "https://www.eventbriteapi.com/v3/events/search/"
    params = {
        "q": query,
        "location.address": location,
        "start_date.range_start": (utc_now()).strftime("%Y-%m-%dT00:00:00Z"),
        "start_date.range_end": (utc_now() + timedelta(days=days_ahead)).strftime("%Y-%m-%dT23:59:59Z"),
        "expand": "venue",
    }
    try:
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {EVENTBRITE_TOKEN}"}, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = []
        for ev in data.get("events", []):
            items.append({
                "name": ev.get("name",{}).get("text"),
                "start": ev.get("start",{}).get("local"),
                "venue": (ev.get("venue") or {}).get("name"),
                "address": (ev.get("venue") or {}).get("address",{}).get("localized_address_display"),
                "url": ev.get("url"),
            })
        df = pd.DataFrame(items)
        if not df.empty and "start" in df:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def rss_items(rss_url: str, max_items: int = 20) -> pd.DataFrame:
    if not feedparser:
        return pd.DataFrame()
    try:
        feed = feedparser.parse(rss_url)
        rows = []
        for e in (feed.entries or [])[:max_items]:
            rows.append({
                "title": getattr(e, "title", None),
                "link": getattr(e, "link", None),
                "published": pd.to_datetime(getattr(e, "published", None), errors="coerce"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# Miami-Dade zoning / infrastructure monitors (example ArcGIS source; swap with your own)
MD_ZONING_FEATURESERVER = "https://services.arcgis.com/LBbVDC0hKPAnLRpO/ArcGIS/rest/services/Miami_Dade_Zoning_Phillips/FeatureServer"
LAYER_ZONING = 12


def zoning_categories_changed(days_back: int = 14) -> pd.DataFrame:
    """Example: return distinct ZONE/ZONE_DESC present now. (ArcGIS may not offer change history)."""
    data = arcgis_query(MD_ZONING_FEATURESERVER, LAYER_ZONING, {
        "outFields": "ZONE,ZONE_DESC", "returnDistinctValues": True, "returnGeometry": False
    })
    feats = (data or {}).get("features", [])
    rows = [{"ZONE": f.get("attributes",{}).get("ZONE"), "ZONE_DESC": f.get("attributes",{}).get("ZONE_DESC")} for f in feats]
    df = pd.DataFrame(rows).dropna().drop_duplicates().sort_values(["ZONE","ZONE_DESC"]).reset_index(drop=True)
    return df

# ---------------------------
# AI helpers
# ---------------------------

def ai_suggest_priorities(context: str = "", objectives: List[str] = None) -> List[str]:
    objectives = objectives or [
        "Call 3 expired listings",
        "Host a virtual tour for your best listing",
        "Post about current mortgage rates on Instagram"
    ]
    if not (OPENAI_API_KEY and openai):
        return objectives
    prompt = (
        "You are a real estate sales assistant. Given the context and goals, list 3 short, concrete actions for today.\n"
        f"Context: {context}\nGoals: lead generation, listing acquisition, nurturing sphere.\nReturn a numbered list."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=160,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        lines = [ln.strip(" -") for ln in text.splitlines() if ln.strip()]
        return lines[:5]
    except Exception:
        return objectives


def ai_generate_content(topic: str, audience: str = "buyers and sellers") -> Dict[str, str]:
    base = {
        "post": f"Quick tip about {topic}: here are 3 things {audience} should know today.",
        "blog": f"Why {topic} can be a hidden advantage this season — 5 fast facts.",
        "email": f"Subject: {topic} — what it means for you\n\nHi there, here’s a short update on {topic}.",
    }
    if not (OPENAI_API_KEY and openai):
        return base
    prompt = (
        "Create a 120-character social post, a 200-word blog outline, and a short email newsletter on the topic.\n"
        f"Topic: {topic}. Audience: {audience}. Style: clear, positive, professional."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        # naive split
        return {**base, "llm": text}
    except Exception:
        return base


def ai_filter_leads(df: pd.DataFrame, criteria: str) -> pd.DataFrame:
    """If OpenAI configured, score/rank; otherwise keyword filter."""
    if df.empty:
        return df
    if not (OPENAI_API_KEY and openai):
        # cheap keyword filter
        kw = [k.strip().lower() for k in re.split(r",|/|;", criteria) if k.strip()]
        if not kw:
            return df
        mask = df.apply(lambda r: any(k in (str(r.get("note","")) + " " + str(r.get("source",""))).lower() for k in kw), axis=1)
        return df[mask]
    # LLM-based scoring
    rows = []
    for _, r in df.iterrows():
        text = f"Lead: name={r.get('name')} note={r.get('note')} source={r.get('source')}"
        prompt = f"Score 0-10 for match to: {criteria}. Return ONLY the number.\n{text}"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4,
            )
            score = float(re.findall(r"\d+(?:\.\d+)?", resp["choices"][0]["message"]["content"] or "0")[0])
        except Exception:
            score = 0
        rr = r.to_dict()
        rr["score"] = score
        rows.append(rr)
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False)

# ---------------------------
# UI — Sidebar
# ---------------------------

st.title("📡 Real Estate Intel — Market Radar & Lead Gen")
st.caption("Prototype — plug in API tokens in secrets to activate live feeds. Demo data shown otherwise.")

with st.sidebar:
    st.header("Targeting")
    target_areas = st.text_input("Target areas (comma separated)", "Doral, Miami, Hialeah")
    buyer_filters = st.text_input("AI filters (comma separated)", "renters turning 30, relocating tech workers, VA buyers")
    price_max = st.number_input("Price cap ($)", value=800000, step=5000)
    st.divider()
    st.header("Integrations status")
    st.write({
        "FRED_API_KEY": bool(FRED_API_KEY),
        "EVENTBRITE_TOKEN": bool(EVENTBRITE_TOKEN),
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "SLACK_WEBHOOK": bool(SLACK_WEBHOOK_URL),
        "PUSHOVER": bool(PUSHOVER_TOKEN and PUSHOVER_USER),
    })

# ---------------------------
# Tabs
# ---------------------------

tab_leads, tab_market, tab_content, tab_geo, tab_settings = st.tabs([
    "👥 Leads", "📈 Market Watch", "📝 Content Studio", "📍 Geofence", "⚙️ Settings"
])

with tab_leads:
    st.subheader("Lead discovery & filtering")
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown("**Paste or upload leads** (CSV columns suggested: name, email, source, note)")
        lead_text = st.text_area("Paste leads (one per line, CSV-ish)", height=140, placeholder="Alex, alex@example.com, Instagram, Renter turning 30; wants 2/2 in Doral\n...")
        uploaded = st.file_uploader("…or upload CSV", type=["csv"])
        df_leads = pd.DataFrame()
        if lead_text.strip():
            rows = []
            for line in lead_text.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    rows.append({"name": parts[0], "email": parts[1], "source": parts[2] if len(parts)>2 else "Manual", "note": ",".join(parts[3:])})
            df_leads = pd.DataFrame(rows)
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                df_leads = pd.concat([df_leads, df_up], ignore_index=True)
            except Exception as e:
                st.error(f"CSV read error: {e}")
        if df_leads.empty:
            df_leads = DEMO_LEADS.copy()
        st.dataframe(df_leads, use_container_width=True)
    with c2:
        st.markdown("**AI filter**")
        crit = st.text_input("Filter criteria", buyer_filters)
        if st.button("Apply filter"):
            df_filtered = ai_filter_leads(df_leads, crit)
            st.dataframe(df_filtered, use_container_width=True)
            st.download_button("⬇️ Download filtered leads (CSV)", df_filtered.to_csv(index=False).encode("utf-8"), file_name="filtered_leads.csv", mime="text/csv")
        st.markdown("---")
        st.markdown("**Auto‑add from portals**")
        st.caption("Use Zapier to forward Zillow/Realtor lead emails into Google Sheets; read that Sheet here.")
        st.text_input("Google Sheet URL (CSV export)", key="sheet_url")
        if st.button("Fetch Sheet"):
            url = st.session_state.get("sheet_url")
            try:
                df_sheet = pd.read_csv(url)
                st.success(f"Loaded {len(df_sheet)} rows from sheet")
                st.dataframe(df_sheet.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Sheet fetch error: {e}")

    st.markdown("---")
    st.subheader("New listings & price drops (demo or your feed)")
    listings = DEMO_LISTINGS.copy()
    areas = [a.strip().lower() for a in target_areas.split(",") if a.strip()]
    if areas:
        listings = listings[listings["address"].str.lower().str.contains("|".join([re.escape(a) for a in areas]))]
    listings = listings[listings["price"] <= price_max]
    st.dataframe(listings, use_container_width=True)
    st.download_button("⬇️ Download listings (CSV)", listings.to_csv(index=False).encode("utf-8"), file_name="listings.csv", mime="text/csv")

with tab_market:
    st.subheader("Mortgage rates (FRED: 30‑year fixed)")
    df_rates = fred_mortgage_rate_series()
    if not df_rates.empty:
        st.line_chart(df_rates.set_index("date")["value"])
    else:
        st.info("No rate data available (check FRED_API_KEY secret)")

    st.markdown("---")
    st.subheader("Local events (Eventbrite)")
    q = st.text_input("Search events", "open house OR festival OR community meeting")
    loc = st.text_input("Location", "Miami, FL")
    if st.button("Search events"):
        df_ev = eventbrite_events(q, loc)
        if not df_ev.empty:
            st.dataframe(df_ev, use_container_width=True)
        else:
            st.info("No events or missing EVENTBRITE_TOKEN.")

    st.markdown("---")
    st.subheader("Zoning / Infrastructure monitors (example: Miami‑Dade)")
    df_z = zoning_categories_changed()
    if not df_z.empty:
        st.dataframe(df_z, use_container_width=True, hide_index=True)
    else:
        st.info("ArcGIS service unavailable or no data.")

with tab_content:
    st.subheader("AI Daily Priorities")
    context = st.text_area("Context (today's listings, appointments, goals)")
    if st.button("Suggest my top 3 actions"):
        items = ai_suggest_priorities(context)
        for i, it in enumerate(items, 1):
            st.write(f"{i}. {it}")

    st.markdown("---")
    st.subheader("Content Studio — social post, blog idea, email")
    topic = st.text_input("Topic", "Why winter can be a secret advantage for buyers")
    audience = st.text_input("Audience", "first‑time buyers in Miami‑Dade")
    if st.button("Generate content"):
        out = ai_generate_content(topic, audience)
        st.markdown("**Social post**")
        st.code(out.get("post"))
        st.markdown("**Blog idea**")
        st.code(out.get("blog"))
        st.markdown("**Email/newsletter**")
        st.code(out.get("email"))
        if out.get("llm"):
            st.markdown("**LLM expanded draft**")
            st.write(out.get("llm"))

with tab_geo:
    st.subheader("Geofence check (manual)")
    st.caption("Browser location compared to nearby hot listings (demo coordinates). For true push notifications, use worker.py with Pushover/Slack and a mobile tracker.")
    dist_m = st.slider("Alert radius (meters)", 100, 5000, 800, step=50)
    myloc = None
    if streamlit_geolocation:
        myloc = streamlit_geolocation()
    else:
        st.info("Install streamlit-geolocation to enable browser GPS: add 'streamlit-geolocation' to requirements.txt")
    if myloc and myloc.get("latitude") and myloc.get("longitude"):
        lat = myloc["latitude"]; lon = myloc["longitude"]
        st.success(f"Your location: {lat:.5f}, {lon:.5f}")
        df = DEMO_LISTINGS.copy()
        df["distance_m"] = df.apply(lambda r: haversine_m(lat, lon, r["lat"], r["lon"]), axis=1)
        near = df[df["distance_m"] <= dist_m].sort_values("distance_m")
        if not near.empty:
            st.warning("Nearby hot listings:")
            st.dataframe(near[["id","address","price","distance_m","url"]], use_container_width=True)
        else:
            st.write("No listings within the selected radius.")
    else:
        st.caption("Allow location access and click the geolocation button (if available).")

with tab_settings:
    st.subheader("Notifications & worker")
    st.markdown("- **Slack**: set `SLACK_WEBHOOK_URL` in secrets for alert pings.\n- **Pushover**: set `PUSHOVER_TOKEN` and `PUSHOVER_USER` for mobile push.\n- **Email**: set `SENDGRID_API_KEY` + `ALERT_EMAIL_TO`.")
    st.markdown("- **GitHub Actions**: use the provided workflow in README to run `worker.py` hourly for true real‑time alerts.")
    st.subheader("APIs")
    st.markdown("- **Mortgage rates (FRED)**: set `FRED_API_KEY`.\n- **Eventbrite**: set `EVENTBRITE_TOKEN`.\n- **OpenAI**: set `OPENAI_API_KEY` for AI filters, priorities, and content.")

st.caption("© Real Estate Intel — demo app. Verify third‑party TOS; prefer official APIs over scraping.")

# =============================
# File: worker.py
# Scheduled alerts (run via GitHub Actions). Reads same env/secrets as app.py
# =============================

worker_py = r"""
import os
import json
import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

FRED_API_KEY = os.getenv("FRED_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_TOKEN")

DEMO_LISTINGS = pd.DataFrame([
    {"id":"L-1001","address":"123 Main St, Doral, FL","price":650000,"status":"new","url":"https://example.com/123main"},
    {"id":"L-1002","address":"456 Lake Ave, Miami, FL","price":489000,"status":"price_drop","url":"https://example.com/456lake"},
])


def notify_slack(text: str):
    if not SLACK_WEBHOOK_URL:
        print("[SLACK disabled]", text)
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=15)
    except Exception as e:
        print("Slack error", e)


def notify_pushover(title: str, message: str):
    if not (PUSHOVER_TOKEN and PUSHOVER_USER):
        print("[Pushover disabled]", title, message)
        return
    try:
        requests.post("https://api.pushover.net/1/messages.json", data={
            "token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "title": title, "message": message
        }, timeout=15)
    except Exception as e:
        print("Pushover error", e)


def fred_rates():
    if not FRED_API_KEY:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id":"MORTGAGE30US","api_key":FRED_API_KEY,"file_type":"json","observation_start":"2023-01-01"}
    try:
        js = requests.get(url, params=params, timeout=20).json()
        obs = js.get("observations", [])
        if not obs:
            return None
        last_val = None
        for o in reversed(obs):
            try:
                last_val = float(o.get("value")); break
            except Exception:
                continue
        return last_val
    except Exception:
        return None


def main():
    # Example: alert on new/price_drop listings (replace with real source)
    new_items = DEMO_LISTINGS[DEMO_LISTINGS["status"].isin(["new","price_drop"])].to_dict("records")
    for it in new_items:
        msg = f"{it['status'].upper()}: {it['address']} at ${it['price']:,} — {it['url']}"
        notify_slack(msg)
        notify_pushover("Listing Alert", msg)
    rate = fred_rates()
    if rate:
        notify_slack(f"30yr mortgage (FRED) latest: {rate:.2f}%")

if __name__ == "__main__":
    main()
"""

# =============================
# File: requirements.txt
# =============================

requirements_txt = r"""
streamlit==1.37.1
pandas==2.2.2
requests==2.32.3
feedparser==6.0.11
python-dateutil==2.9.0.post0
streamlit-geolocation==0.3.1
openai==0.28.1
"""

# =============================
# File: .streamlit/secrets.toml.example
# =============================

secrets_toml = r"""
# Copy to .streamlit/secrets.toml and fill in values (or set as Streamlit Cloud Secrets)
FRED_API_KEY = "your_fred_api_key"
OPENAI_API_KEY = "your_openai_api_key"
EVENTBRITE_TOKEN = "your_eventbrite_token"
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
PUSHOVER_TOKEN = "your_pushover_app_token"
PUSHOVER_USER = "your_pushover_user_key"
SENDGRID_API_KEY = "your_sendgrid_api_key"
ALERT_EMAIL_TO = "you@example.com"
"""

# =============================
# File: README.md (excerpt)
# =============================

readme_md = r"""
# Real Estate Intel — Market Radar & Lead Gen (Streamlit)

A deployable Streamlit app + optional GitHub Actions worker for alerts.

## Quick Start

1. **Create repo** and add these files: `app.py`, `requirements.txt`, `.streamlit/secrets.toml` (optional), `worker.py`.
2. **Run locally**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. **Streamlit Cloud**: Deploy the repo; add secrets (FRED_API_KEY, EVENTBRITE_TOKEN, OPENAI_API_KEY, etc.).

## Alerts (GitHub Actions)

Create `.github/workflows/alerts.yml`:
```yaml
name: Alerts
on:
  schedule:
    - cron: "0 * * * *"  # hourly
  workflow_dispatch: {}
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python - <<'PY'
import os
from pathlib import Path
code = Path('worker.py').read_text()
exec(code, {})
PY
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
          PUSHOVER_TOKEN: ${{ secrets.PUSHOVER_TOKEN }}
          PUSHOVER_USER: ${{ secrets.PUSHOVER_USER }}
          EVENTBRITE_TOKEN: ${{ secrets.EVENTBRITE_TOKEN }}
```

## Data Sources
- **Listings/price drops**: Use your MLS vendor feed or portal webhooks → Google Sheets → CSV URL.
- **Events**: Eventbrite API (token) or add RSS feeds in `rss_items()`.
- **Mortgage rates**: FRED `MORTGAGE30US`.
- **Zoning/infrastructure**: Example ArcGIS query; replace with your county/service.
- **AI**: OpenAI for content & prioritization (optional).

> Respect third‑party Terms of Service. Prefer official APIs and webhooks over scraping.
"""

# Render helper to let users download the extra files directly from the app
with st.expander("📦 Export supporting files (copy these to your repo)"):
    st.download_button("📄 Download worker.py", worker_py, file_name="worker.py")
    st.download_button("🧾 Download requirements.txt", requirements_txt, file_name="requirements.txt")
    st.download_button("🔐 Download secrets template", secrets_toml, file_name="secrets.toml.example")
    st.download_button("📘 Download README excerpt", readme_md, file_name="README.md")

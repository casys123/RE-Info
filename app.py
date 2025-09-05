#!/usr/bin/env python3
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Union

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import requests
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Configuration
# -----------------------------
app.config["MONGO_URI"] = os.getenv("MONGODB_URI", "mongodb://localhost:27017/realestate")
app.config["EVENTBRITE_TOKEN"] = os.getenv("EVENTBRITE_TOKEN")
app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-key")

# -----------------------------
# Database setup
# -----------------------------
client = MongoClient(app.config["MONGO_URI"])

# Prefer default DB from URI if present
try:
    db = client.get_default_database()
except Exception:
    # Fallback to explicit DB name if URI had none
    db_name = os.getenv("MONGODB_DBNAME", "realestate")
    db = client[db_name]

leads_collection = db.leads
tasks_collection = db.tasks
events_collection = db.events
market_data_collection = db.market_data
users_collection = db.users

# -----------------------------
# Logging configuration
# -----------------------------
handler = RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# -----------------------------
# Helpers / Utilities
# -----------------------------
def _dt_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO 8601 string (UTC, with 'Z')."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def serialize_value(val: Any) -> Any:
    """Recursively serialize Mongo/Datetime values to JSON-safe types."""
    if isinstance(val, ObjectId):
        return str(val)
    if isinstance(val, datetime):
        return _dt_to_iso(val)
    if isinstance(val, list):
        return [serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    return val


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return serialize_value(doc)


def parse_iso_datetime(s: str) -> datetime:
    """
    Parse ISO 8601 strings like "2025-09-05T12:34:56Z" or "2025-09-05".
    Returns naive UTC datetime.
    """
    s = s.strip()
    # Accept plain dates as local midnight UTC
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    # Accept trailing Z
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    # Convert to naive UTC for Mongo
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def validate_lead_data(data: Dict[str, Any]):
    required_fields = ["name", "contact"]
    if not isinstance(data, dict) or not all(field in data for field in required_fields):
        raise ValueError("Missing required lead fields: name, contact")


def get_user_id_from_token():
    # NOTE: placeholder; replace with real JWT verification if needed
    return request.headers.get("X-User-ID", "default_user")


def get_default_tasks() -> List[Dict[str, Any]]:
    return [
        {"title": "Follow up with 5 hot leads", "category": "leads", "priority": 1, "estimated_time": "45m"},
        {"title": "Research new listings in target area", "category": "market", "priority": 2, "estimated_time": "30m"},
        {"title": "Post market update on social media", "category": "marketing", "priority": 3, "estimated_time": "20m"},
        {"title": "Attend local networking event", "category": "networking", "priority": 4, "estimated_time": "60m"},
        {"title": "Review weekly performance metrics", "category": "admin", "priority": 5, "estimated_time": "30m"},
    ]


# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return jsonify({"status": "Real Estate Agent Companion API", "version": "1.0.1"})


# -----------------------------
# Lead Management Endpoints
# -----------------------------
@app.route("/api/leads", methods=["GET", "POST"])
def leads():
    user_id = get_user_id_from_token()

    if request.method == "GET":
        # Pagination & filters
        try:
            page = max(int(request.args.get("page", 1)), 1)
            limit = max(min(int(request.args.get("limit", 10)), 100), 1)
        except ValueError:
            return bad_request("Invalid pagination parameters")
        status = request.args.get("status")

        query: Dict[str, Any] = {"user_id": user_id}
        if status:
            query["status"] = status

        cursor = leads_collection.find(query).skip((page - 1) * limit).limit(limit)
        docs = [serialize_doc(doc) for doc in cursor]
        total = leads_collection.count_documents(query)

        return jsonify({"page": page, "limit": limit, "total": total, "data": docs})

    # POST
    try:
        data = request.get_json(force=True, silent=False)
        validate_lead_data(data)

        notes = data.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        elif not isinstance(notes, list):
            notes = []

        follow_up_date = data.get("follow_up_date")
        if isinstance(follow_up_date, str) and follow_up_date.strip():
            try:
                follow_up_date = parse_iso_datetime(follow_up_date)
            except Exception:
                return bad_request("Invalid follow_up_date (use ISO 8601, e.g., 2025-09-05 or 2025-09-05T14:30:00Z)")
        else:
            follow_up_date = None

        now = datetime.utcnow()
        lead = {
            "name": data["name"],
            "contact": data["contact"],
            "source": data.get("source", "unknown"),
            "status": data.get("status", "new"),
            "notes": notes,
            "follow_up_date": follow_up_date,
            "last_contacted": now,
            "user_id": user_id,
            "created_at": now,
        }

        result = leads_collection.insert_one(lead)
        lead["_id"] = result.inserted_id
        app.logger.info(f'New lead created: {lead["name"]}')
        return jsonify(serialize_doc(lead)), 201

    except ValueError as e:
        return bad_request(e)
    except Exception as e:
        app.logger.exception("Error creating lead")
        return internal_error(e)


# -----------------------------
# Task Management Endpoints
# -----------------------------
@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    user_id = get_user_id_from_token()
    date_filter = request.args.get("date", "today")

    try:
        if date_filter == "today":
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            query = {"user_id": user_id, "due_date": {"$gte": start_date, "$lt": end_date}}
        elif date_filter == "upcoming":
            query = {"user_id": user_id, "due_date": {"$gte": datetime.utcnow()}}
        else:
            return bad_request("Invalid date filter (use 'today' or 'upcoming')")

        tasks = list(tasks_collection.find(query).sort("priority", 1))
        tasks = [serialize_doc(t) for t in tasks]
        return jsonify({"count": len(tasks), "data": tasks})

    except Exception as e:
        app.logger.exception("Error fetching tasks")
        return internal_error(e)


@app.route("/api/tasks/generate", methods=["POST"])
def generate_tasks():
    # If no OpenAI key, return defaults but indicate not configured
    if not app.config["OPENAI_API_KEY"]:
        return jsonify({"error": "OpenAI integration not configured", "tasks": get_default_tasks()}), 200

    prompt = (
        "Generate 5 specific daily tasks for a real estate agent to increase business. "
        "Include lead follow-ups, market research, networking, and personal development activities. "
        "Return ONLY a JSON array with items having: title, category, priority (1-5), estimated_time."
    )

    headers = {"Authorization": f'Bearer {app.config["OPENAI_API_KEY"]}', "Content-Type": "application/json"}

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
            timeout=30,
        )
        if resp.status_code != 200:
            app.logger.error("OpenAI error: %s %s", resp.status_code, resp.text)
            return jsonify({"error": "AI task generation failed", "tasks": get_default_tasks()}), 200

        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Try to parse JSON directly; if it fails, try to extract from code fences
        tasks_json: Union[List[Dict[str, Any]], None] = None
        try:
            tasks_json = json.loads(content)
        except Exception:
            # Attempt to pull JSON from ``` blocks
            if "```" in content:
                parts = content.split("```")
                for block in parts:
                    block = block.strip()
                    if block.startswith("json"):
                        block = block[4:].strip()
                    try:
                        tasks_json = json.loads(block)
                        break
                    except Exception:
                        continue

        if not isinstance(tasks_json, list):
            raise ValueError("Model did not return a valid JSON array")

        # Optional: sanitize/clip fields
        cleaned = []
        for item in tasks_json:
            if not isinstance(item, dict):
                continue
            cleaned.append(
                {
                    "title": str(item.get("title", "")).strip()[:200],
                    "category": str(item.get("category", "")).strip()[:50],
                    "priority": int(item.get("priority", 3)) if str(item.get("priority", "")).isdigit() else 3,
                    "estimated_time": str(item.get("estimated_time", "30m")).strip()[:20],
                }
            )

        return jsonify({"tasks": cleaned}), 200

    except Exception as e:
        app.logger.exception("AI task generation failed")
        return jsonify({"error": "AI task generation failed", "tasks": get_default_tasks()}), 200


# -----------------------------
# Market Data Endpoints
# -----------------------------
@app.route("/api/market", methods=["GET"])
def get_market_data():
    location = request.args.get("location", "New York")
    # days param accepted for future use; currently unused in simulation
    _ = int(request.args.get("days", 30))

    try:
        # Check cache (valid for 6 hours)
        cache = market_data_collection.find_one(
            {"location": location, "updated_at": {"$gte": datetime.utcnow() - timedelta(hours=6)}}
        )
        if cache and "data" in cache:
            return jsonify(serialize_doc(cache["data"]))

        # Simulate data (replace with MLS/3rd-party integrations)
        market_data = {
            "location": location,
            "average_price": simulate_market_price(location),
            "days_on_market": simulate_days_on_market(location),
            "inventory": simulate_inventory(location),
            "updated_at": _dt_to_iso(datetime.utcnow()),
        }

        market_data_collection.update_one(
            {"location": location}, {"$set": {"data": market_data, "updated_at": datetime.utcnow()}}, upsert=True
        )

        return jsonify(market_data)

    except Exception as e:
        app.logger.error(f"Market data error: {str(e)}")
        return internal_error("Failed to fetch market data")


def simulate_market_price(location: str) -> float:
    # In real implementation, connect to MLS or vendor API
    base_prices = {"New York": 850000, "Los Angeles": 750000, "Chicago": 350000}
    base = base_prices.get(location, 500000)
    # Simple day-of-month seasonal factor
    factor = 0.9 + 0.2 * (datetime.utcnow().day / 31)
    return round(base * factor, 2)


def simulate_days_on_market(location: str) -> int:
    baseline = {"New York": 45, "Los Angeles": 40, "Chicago": 55}
    base = baseline.get(location, 50)
    jitter = (datetime.utcnow().day % 7) - 3  # -3..+3
    return max(5, base + jitter)


def simulate_inventory(location: str) -> int:
    baseline = {"New York": 12000, "Los Angeles": 9000, "Chicago": 6000}
    base = baseline.get(location, 7000)
    jitter = (datetime.utcnow().day % 11) - 5  # -5..+5
    return max(100, base + jitter)


# -----------------------------
# Event Management Endpoints
# -----------------------------
@app.route("/api/events", methods=["GET", "POST"])
def events():
    if request.method == "GET":
        return get_events()
    elif request.method == "POST":
        return create_event()


def get_events():
    location = request.args.get("location", "New York")
    try:
        # Upcoming manual events
        manual_events_cursor = (
            events_collection.find({"date": {"$gte": datetime.utcnow()}})
            .sort("date", 1)
        )
        manual_events = [serialize_doc(doc) for doc in manual_events_cursor]

        # Eventbrite events
        eventbrite_events: List[Dict[str, Any]] = []
        if app.config["EVENTBRITE_TOKEN"]:
            eventbrite_events = fetch_eventbrite_events(location)

        response = {
            "manual_events": manual_events,
            "eventbrite_events": eventbrite_events,
            "eventbrite_configured": bool(app.config["EVENTBRITE_TOKEN"]),
        }
        return jsonify(response)

    except Exception as e:
        app.logger.exception("Event fetch error")
        return internal_error("Failed to fetch events")


def fetch_eventbrite_events(location: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(
            "https://www.eventbriteapi.com/v3/events/search/",
            params={
                "location.address": location,
                "categories": "101,110",  # Business & Real Estate
                "sort_by": "date",
                "expand": "venue",        # IMPORTANT: we access e['venue']['name'] below
                "page_size": 25,
            },
            headers={"Authorization": f'Bearer {app.config["EVENTBRITE_TOKEN"]}'},
            timeout=20,
        )
        if resp.status_code != 200:
            app.logger.warning("Eventbrite non-200: %s %s", resp.status_code, resp.text)
            return []

        events = resp.json().get("events", [])
        formatted = []
        for e in events:
            name = (e.get("name") or {}).get("text")
            description = (e.get("description") or {}).get("text")
            start_local = (e.get("start") or {}).get("local")
            end_local = (e.get("end") or {}).get("local")
            url = e.get("url")
            venue_name = (e.get("venue") or {}).get("name")
            logo_url = (e.get("logo") or {}).get("url") if e.get("logo") else None

            formatted.append(
                {
                    "id": e.get("id"),
                    "name": name,
                    "description": description,
                    "url": url,
                    "start": start_local,
                    "end": end_local,
                    "venue": venue_name,
                    "logo": logo_url,
                }
            )
        return formatted

    except Exception as e:
        app.logger.warning(f"Eventbrite fetch failed: {str(e)}")
        return []


def create_event():
    try:
        data = request.get_json(force=True, silent=False)
        required = ["title", "date", "location"]
        if not all(k in data for k in required):
            return bad_request("Missing required fields: title, date, location")

        try:
            parsed_date = parse_iso_datetime(str(data["date"]))
        except Exception:
            return bad_request("Invalid date format; use ISO 8601 (e.g., 2025-09-05T14:30:00Z)")

        event = {
            "title": data["title"],
            "description": data.get("description", ""),
            "date": parsed_date,
            "location": data["location"],
            "created_by": get_user_id_from_token(),
            "created_at": datetime.utcnow(),
        }

        result = events_collection.insert_one(event)
        event["_id"] = result.inserted_id
        return jsonify(serialize_doc(event)), 201

    except Exception as e:
        app.logger.exception("Create event failed")
        return internal_error(e)


# -----------------------------
# Authentication (simplified)
# -----------------------------
@app.route("/api/auth/login", methods=["POST"])
def login():
    # In production, implement proper authentication + JWT issuance
    return jsonify({"token": "sample-jwt-token", "user_id": "demo-user"})


# -----------------------------
# Health Check
# -----------------------------
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": _dt_to_iso(datetime.utcnow())})


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)

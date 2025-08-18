#!/usr/bin/env python3
import os
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import requests
import logging
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MONGO_URI'] = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/realestate')
app.config['EVENTBRITE_TOKEN'] = os.getenv('EVENTBRITE_TOKEN')
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key')

# Database setup
client = MongoClient(app.config['MONGO_URI'])
db = client.get_database()
leads_collection = db.leads
tasks_collection = db.tasks
events_collection = db.events
market_data_collection = db.market_data
users_collection = db.users

# Logging configuration
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Error Handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

# Utility Functions
def validate_lead_data(data):
    required_fields = ['name', 'contact']
    if not all(field in data for field in required_fields):
        raise ValueError('Missing required lead fields')

def get_user_id_from_token():
    # In a real implementation, decode JWT from Authorization header
    return request.headers.get('X-User-ID', 'default_user')

# Routes
@app.route('/')
def index():
    return jsonify({'status': 'Real Estate Agent Companion API', 'version': '1.0.0'})

# Lead Management Endpoints
@app.route('/api/leads', methods=['GET', 'POST'])
def leads():
    user_id = get_user_id_from_token()
    
    if request.method == 'GET':
        # Get paginated leads with optional filters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        status = request.args.get('status')
        
        query = {'user_id': user_id}
        if status:
            query['status'] = status
            
        leads = list(leads_collection.find(query).skip((page-1)*limit).limit(limit))
        for lead in leads:
            lead['_id'] = str(lead['_id'])
        
        return jsonify({
            'page': page,
            'limit': limit,
            'total': leads_collection.count_documents(query),
            'data': leads
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            validate_lead_data(data)
            
            lead = {
                'name': data['name'],
                'contact': data['contact'],
                'source': data.get('source', 'unknown'),
                'status': data.get('status', 'new'),
                'notes': data.get('notes', []),
                'follow_up_date': data.get('follow_up_date'),
                'last_contacted': datetime.utcnow(),
                'user_id': user_id,
                'created_at': datetime.utcnow()
            }
            
            result = leads_collection.insert_one(lead)
            lead['_id'] = str(result.inserted_id)
            
            app.logger.info(f'New lead created: {lead["name"]}')
            return jsonify(lead), 201
        
        except ValueError as e:
            return bad_request(e)
        except Exception as e:
            return internal_error(e)

# Task Management Endpoints
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    user_id = get_user_id_from_token()
    date_filter = request.args.get('date', 'today')
    
    try:
        if date_filter == 'today':
            start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            query = {
                'user_id': user_id,
                'due_date': {'$gte': start_date, '$lt': end_date}
            }
        elif date_filter == 'upcoming':
            query = {
                'user_id': user_id,
                'due_date': {'$gte': datetime.utcnow()}
            }
        else:
            return bad_request('Invalid date filter')
        
        tasks = list(tasks_collection.find(query).sort('priority', 1))
        for task in tasks:
            task['_id'] = str(task['_id'])
        
        return jsonify(tasks)
    
    except Exception as e:
        return internal_error(e)

@app.route('/api/tasks/generate', methods=['POST'])
def generate_tasks():
    if not app.config['OPENAI_API_KEY']:
        return jsonify({
            'error': 'OpenAI integration not configured',
            'tasks': get_default_tasks()
        }), 200
    
    try:
        prompt = """Generate 5 specific daily tasks for a real estate agent to increase business. 
        Include lead follow-ups, market research, networking, and personal development activities.
        Return as JSON array with fields: title, category, priority (1-5), estimated_time."""
        
        headers = {
            'Authorization': f'Bearer {app.config["OPENAI_API_KEY"]}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json={
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7
            }
        )
        
        tasks = response.json()['choices'][0]['message']['content']
        return jsonify(tasks), 200
    
    except Exception as e:
        app.logger.error(f'AI task generation failed: {str(e)}')
        return jsonify({
            'error': 'AI task generation failed',
            'tasks': get_default_tasks()
        }), 200

def get_default_tasks():
    return [
        {"title": "Follow up with 5 hot leads", "category": "leads", "priority": 1},
        {"title": "Research new listings in target area", "category": "market", "priority": 2},
        {"title": "Post market update on social media", "category": "marketing", "priority": 3},
        {"title": "Attend local networking event", "category": "networking", "priority": 4},
        {"title": "Review weekly performance metrics", "category": "admin", "priority": 5}
    ]

# Market Data Endpoints
@app.route('/api/market', methods=['GET'])
def get_market_data():
    location = request.args.get('location', 'New York')
    days = int(request.args.get('days', 30))
    
    try:
        # Get cached market data
        cache = market_data_collection.find_one({
            'location': location,
            'updated_at': {'$gte': datetime.utcnow() - timedelta(hours=6)}
        })
        
        if cache:
            return jsonify(cache['data'])
        
        # If no recent cache, fetch new data (simulated here)
        market_data = {
            'location': location,
            'average_price': simulate_market_price(location),
            'days_on_market': simulate_days_on_market(location),
            'inventory': simulate_inventory(location),
            'updated_at': datetime.utcnow()
        }
        
        # Cache the data
        market_data_collection.update_one(
            {'location': location},
            {'$set': {'data': market_data, 'updated_at': datetime.utcnow()}},
            upsert=True
        )
        
        return jsonify(market_data)
    
    except Exception as e:
        app.logger.error(f'Market data error: {str(e)}')
        return internal_error('Failed to fetch market data')

def simulate_market_price(location):
    # In real implementation, connect to MLS API
    base_prices = {'New York': 850000, 'Los Angeles': 750000, 'Chicago': 350000}
    return base_prices.get(location, 500000) * (0.9 + 0.2 * (datetime.now().day / 31))

# Event Management Endpoints
@app.route('/api/events', methods=['GET', 'POST'])
def events():
    if request.method == 'GET':
        return get_events()
    elif request.method == 'POST':
        return create_event()

def get_events():
    location = request.args.get('location', 'New York')
    try:
        # Get manual events
        manual_events = list(events_collection.find({
            'date': {'$gte': datetime.utcnow()}
        }).sort('date', 1))
        
        # Get Eventbrite events if configured
        eventbrite_events = []
        if app.config['EVENTBRITE_TOKEN']:
            eventbrite_events = fetch_eventbrite_events(location)
        
        # Format response
        response = {
            'manual_events': [format_event(e) for e in manual_events],
            'eventbrite_events': eventbrite_events,
            'eventbrite_configured': bool(app.config['EVENTBRITE_TOKEN'])
        }
        
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f'Event fetch error: {str(e)}')
        return internal_error('Failed to fetch events')

def fetch_eventbrite_events(location):
    try:
        response = requests.get(
            'https://www.eventbriteapi.com/v3/events/search/',
            params={
                'location.address': location,
                'categories': '101,110',  # Business & Real Estate
                'sort_by': 'date'
            },
            headers={
                'Authorization': f'Bearer {app.config["EVENTBRITE_TOKEN"]}'
            }
        )
        
        events = response.json().get('events', [])
        return [{
            'id': e['id'],
            'name': e['name']['text'],
            'description': e['description']['text'],
            'url': e['url'],
            'start': e['start']['local'],
            'end': e['end']['local'],
            'venue': e['venue']['name'],
            'logo': e['logo']['url'] if e.get('logo') else None
        } for e in events]
    
    except Exception as e:
        app.logger.warning(f'Eventbrite fetch failed: {str(e)}')
        return []

def create_event():
    try:
        data = request.get_json()
        required_fields = ['title', 'date', 'location']
        if not all(field in data for field in required_fields):
            return bad_request('Missing required fields')
        
        event = {
            'title': data['title'],
            'description': data.get('description', ''),
            'date': datetime.fromisoformat(data['date']),
            'location': data['location'],
            'created_by': get_user_id_from_token(),
            'created_at': datetime.utcnow()
        }
        
        result = events_collection.insert_one(event)
        event['_id'] = str(result.inserted_id)
        
        return jsonify(event), 201
    
    except ValueError as e:
        return bad_request('Invalid date format')
    except Exception as e:
        return internal_error(e)

def format_event(event):
    event['_id'] = str(event['_id'])
    event['date'] = event['date'].isoformat()
    return event

# Authentication Endpoints (simplified)
@app.route('/api/auth/login', methods=['POST'])
def login():
    # In production, use proper auth with JWT
    return jsonify({'token': 'sample-jwt-token', 'user_id': 'demo-user'})

# Health Check
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
from transformers import pipeline
from flask import Flask, request, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Base URL of your Spring backend
SPRING_API_BASE_URL = "http://localhost:9080"

# Weights for recommendation factors
WEIGHTS = {
    'distance': 0.3,
    'rating': 0.3,
    'activity': 0.2,
    'popularity': 0.1,
    'sentiment': 0.1
}

def fetch_data(endpoint):
    """Helper function to fetch data from Spring backend without authentication."""
    try:
        response = requests.get(f"{SPRING_API_BASE_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching data from {endpoint}: {e}")
        return []

def calculate_distance(client_location, professional_location):
    """Calculate geodesic distance between two locations."""
    if not client_location or not professional_location:
        return float('inf')
    try:
        client_coords = (client_location.get('latitude'), client_location.get('longitude'))
        prof_coords = (professional_location.get('latitude'), professional_location.get('longitude'))
        return geodesic(client_coords, prof_coords).kilometers
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return float('inf')

def analyze_comments(comments):
    """Analyze sentiment of comments using transformers."""
    if not comments:
        return 0.0
    scores = []
    for comment in comments:
        if comment:
            result = sentiment_analyzer(comment)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            scores.append(score)
    return np.mean(scores) if scores else 0.0

def compute_recommendation_score(client_id):
    """Compute recommendation scores for professionals."""
    # Fetch client data
    client = fetch_data(f"/users/{client_id}")
    if not client:
        return []

    # Fetch all professionals
    professionals = fetch_data("/professionals")
    if not professionals:
        return []

    # Fetch service requests to calculate popularity
    service_requests = fetch_data("/api/service-requests/public")  # Assumes a public endpoint for service requests

    scores = []
    scaler = MinMaxScaler()

    for prof in professionals:
        prof_id = prof.get('id')

        # Distance score (lower distance is better)
        distance = calculate_distance(client.get('localisation'), prof.get('localisation'))
        distance_score = 1 / (1 + distance) if distance != float('inf') else 0.0

        # Rating score
        avg_rating = fetch_data(f"/professionals/{prof_id}/moyenne-avis")
        rating_score = avg_rating / 5.0 if avg_rating else 0.0

        # Activity score (based on publications)
        publications = fetch_data(f"/publications/user/{prof_id}")
        activity_score = sum(pub.get('likes', 0) + len(pub.get('commentaires', [])) for pub in publications)
        activity_score = min(activity_score / 100.0, 1.0)  # Normalize to [0,1]

        # Popularity score (based on service requests)
        request_count = sum(1 for req in service_requests if req.get('professional', {}).get('id') == prof_id)
        popularity_score = min(request_count / 50.0, 1.0)  # Normalize to [0,1]

        # Sentiment score from evaluations
        evaluations = prof.get('evaluations', [])
        comments = [eval.get('comment') for eval in evaluations if eval.get('comment')]
        sentiment_score = analyze_comments(comments)

        # Combine scores with weights
        score = (
            WEIGHTS['distance'] * distance_score +
            WEIGHTS['rating'] * rating_score +
            WEIGHTS['activity'] * activity_score +
            WEIGHTS['popularity'] * popularity_score +
            WEIGHTS['sentiment'] * sentiment_score
        )

        scores.append({
            'professional': prof,
            'score': score
        })

    # Normalize scores
    raw_scores = [[s['score']] for s in scores]
    if raw_scores:
        normalized_scores = scaler.fit_transform(raw_scores).flatten()
        for i, s in enumerate(scores):
            s['score'] = normalized_scores[i]

    # Sort by score and return top professionals
    return sorted(scores, key=lambda x: x['score'], reverse=True)[:10]

@app.route('/recommend/<int:client_id>', methods=['GET'])
def recommend_professionals(client_id):
    """Endpoint to recommend professionals for a client without authentication."""
    try:
        recommendations = compute_recommendation_score(client_id)
        return jsonify([{
            'id': r['professional']['id'],
            'firstName': r['professional'].get('firstName'),
            'lastName': r['professional'].get('lastName'),
            'username': r['professional'].get('username'),
            'score': r['score']
        } for r in recommendations])
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
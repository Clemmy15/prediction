from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date
import random
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="ClemmyTech Bet Generator AI Service",
    description="AI-powered betting prediction microservice",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SlipRequest(BaseModel):
    type: str  # safe, medium, risky, jackpot
    league: Optional[str] = None
    country: Optional[str] = None
    markets: Optional[List[str]] = None

class Match(BaseModel):
    id: str
    home_team: str
    away_team: str
    league: str
    country: str
    match_time: str
    status: str
    odds: Dict[str, Any]
    ai_prediction: Dict[str, Any]
    confidence: float

class Prediction(BaseModel):
    id: str
    type: str
    matches: List[Match]
    total_odds: float
    confidence_avg: float
    created_by: str
    created_at: str

class ResultUpdate(BaseModel):
    results: List[Dict[str, Any]]

# Sample data for demonstration
SAMPLE_LEAGUES = [
    "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1",
    "Champions League", "Europa League", "Eredivisie", "Primeira Liga", "Championship"
]

SAMPLE_COUNTRIES = [
    "England", "Spain", "Italy", "Germany", "France", "Netherlands", "Portugal"
]

SAMPLE_TEAMS = {
    "Premier League": [
        ("Manchester City", "Manchester United"), ("Liverpool", "Chelsea"),
        ("Arsenal", "Tottenham"), ("Newcastle", "Brighton"), ("Aston Villa", "West Ham")
    ],
    "La Liga": [
        ("Real Madrid", "Barcelona"), ("Atletico Madrid", "Sevilla"),
        ("Real Sociedad", "Villarreal"), ("Real Betis", "Valencia")
    ],
    "Serie A": [
        ("Juventus", "Inter Milan"), ("AC Milan", "Napoli"),
        ("Roma", "Lazio"), ("Atalanta", "Fiorentina")
    ],
    "Bundesliga": [
        ("Bayern Munich", "Borussia Dortmund"), ("RB Leipzig", "Bayer Leverkusen"),
        ("Eintracht Frankfurt", "Borussia Monchengladbach")
    ]
}

MARKET_TYPES = {
    "over_under": ["Over 1.5", "Over 2.5", "Under 2.5", "Under 3.5"],
    "double_chance": ["1X", "X2", "12"],
    "btts": ["Yes", "No"],
    "correct_score": ["1-0", "2-1", "2-0", "3-1", "3-2", "0-0", "1-1", "2-2"]
}

def generate_match_id():
    return f"match_{random.randint(100000, 999999)}"

def generate_odds():
    return {
        "over_1.5": round(random.uniform(1.2, 1.8), 2),
        "over_2.5": round(random.uniform(1.4, 2.2), 2),
        "btts_yes": round(random.uniform(1.5, 2.5), 2),
        "btts_no": round(random.uniform(1.3, 2.0), 2),
        "double_chance_1x": round(random.uniform(1.2, 1.6), 2),
        "double_chance_x2": round(random.uniform(1.2, 1.6), 2),
        "correct_score": round(random.uniform(6.0, 15.0), 2)
    }

def generate_ai_prediction(market_type: str):
    predictions = {
        "over_1.5": {"prediction": "Over 1.5", "confidence": random.uniform(0.6, 0.9)},
        "over_2.5": {"prediction": "Over 2.5", "confidence": random.uniform(0.5, 0.8)},
        "btts": {"prediction": random.choice(["Yes", "No"]), "confidence": random.uniform(0.6, 0.85)},
        "double_chance": {"prediction": random.choice(["1X", "X2", "12"]), "confidence": random.uniform(0.7, 0.9)},
        "correct_score": {"prediction": random.choice(["1-0", "2-1", "2-0"]), "confidence": random.uniform(0.3, 0.6)}
    }
    return predictions.get(market_type, {"prediction": "Unknown", "confidence": 0.5})

def get_slip_parameters(slip_type: str):
    """Get parameters based on slip type"""
    params = {
        "safe": {"num_matches": 3, "min_confidence": 0.7, "max_odds": 3.0},
        "medium": {"num_matches": 4, "min_confidence": 0.6, "max_odds": 5.0},
        "risky": {"num_matches": 5, "min_confidence": 0.5, "max_odds": 10.0},
        "jackpot": {"num_matches": 6, "min_confidence": 0.4, "max_odds": 20.0}
    }
    return params.get(slip_type, params["safe"])

@app.get("/")
async def root():
    return {"message": "ClemmyTech Bet Generator AI Service", "status": "running"}

@app.post("/generate-slip", response_model=Prediction)
async def generate_slip(request: SlipRequest):
    """Generate a new AI prediction slip"""
    try:
        params = get_slip_parameters(request.type)
        matches = []
        total_odds = 1.0
        total_confidence = 0.0
        
        # Filter leagues if specified
        available_leagues = [request.league] if request.league else list(SAMPLE_LEAGUES)
        
        for i in range(params["num_matches"]):
            # Select random league
            league = random.choice(available_leagues)
            
            # Get teams for the league
            if league in SAMPLE_TEAMS:
                home_team, away_team = random.choice(SAMPLE_TEAMS[league])
            else:
                home_team = f"Team {i+1}A"
                away_team = f"Team {i+1}B"
            
            # Generate match time (next 1-7 days)
            match_time = datetime.now().replace(hour=random.randint(15, 22), minute=0, second=0, microsecond=0)
            match_time = match_time.replace(day=match_time.day + random.randint(1, 7))
            
            # Select market type
            if request.markets:
                market_type = random.choice(request.markets)
            else:
                market_type = random.choice(list(MARKET_TYPES.keys()))
            
            # Generate odds and prediction
            odds = generate_odds()
            ai_prediction = generate_ai_prediction(market_type)
            
            # Ensure confidence meets minimum requirement
            confidence = max(ai_prediction["confidence"], params["min_confidence"])
            
            # Calculate match odds (simplified)
            match_odds = odds.get(market_type, 2.0)
            total_odds *= match_odds
            
            match = Match(
                id=generate_match_id(),
                home_team=home_team,
                away_team=away_team,
                league=league,
                country=request.country or random.choice(SAMPLE_COUNTRIES),
                match_time=match_time.isoformat(),
                status="upcoming",
                odds=odds,
                ai_prediction=ai_prediction,
                confidence=confidence
            )
            
            matches.append(match)
            total_confidence += confidence
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(matches)
        
        # Ensure total odds doesn't exceed maximum
        if total_odds > params["max_odds"]:
            total_odds = params["max_odds"]
        
        prediction = Prediction(
            id=f"pred_{random.randint(100000, 999999)}",
            type=request.type,
            matches=matches,
            total_odds=round(total_odds, 2),
            confidence_avg=round(avg_confidence, 3),
            created_by="ai_service",
            created_at=datetime.now().isoformat()
        )
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating slip: {str(e)}")

@app.get("/analyze-match")
async def analyze_match(match_id: str):
    """Analyze a single match"""
    try:
        # This is a placeholder - in a real implementation, you would:
        # 1. Fetch match data from a sports API
        # 2. Run AI analysis on historical data
        # 3. Return detailed analysis
        
        analysis = {
            "match_id": match_id,
            "analysis": {
                "home_team_strength": random.uniform(0.3, 0.9),
                "away_team_strength": random.uniform(0.3, 0.9),
                "head_to_head": random.uniform(0.2, 0.8),
                "form_analysis": random.uniform(0.4, 0.9),
                "predicted_score": f"{random.randint(0, 3)}-{random.randint(0, 3)}",
                "confidence": random.uniform(0.6, 0.9)
            },
            "recommendations": [
                "Over 2.5 goals",
                "Both teams to score",
                "Home team win"
            ]
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing match: {str(e)}")

@app.post("/update-results")
async def update_results(request: ResultUpdate):
    """Update results from external API"""
    try:
        # In a real implementation, you would:
        # 1. Process the results data
        # 2. Update your database
        # 3. Retrain models if needed
        
        processed_count = len(request.results)
        
        return {
            "message": f"Successfully processed {processed_count} results",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating results: {str(e)}")

@app.get("/get-results")
async def get_results(from_date: Optional[str] = None, to_date: Optional[str] = None):
    """Get current or past results"""
    try:
        # This is a placeholder - in a real implementation, you would:
        # 1. Query your database for results
        # 2. Filter by date range
        # 3. Return formatted results
        
        results = [
            {
                "match_id": f"match_{i}",
                "home_team": f"Team {i}A",
                "away_team": f"Team {i}B",
                "score": f"{random.randint(0, 3)}-{random.randint(0, 3)}",
                "ai_prediction": "Over 2.5",
                "correct": random.choice([True, False]),
                "date": (datetime.now().replace(day=datetime.now().day - i)).isoformat()
            }
            for i in range(1, 6)
        ]
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.post("/retrain-model")
async def retrain_model():
    """Retrain AI model (Admin only)"""
    try:
        # In a real implementation, you would:
        # 1. Fetch latest data
        # 2. Train new model
        # 3. Validate performance
        # 4. Deploy new model
        
        return {
            "message": "Model retraining initiated",
            "status": "success",
            "estimated_time": "30 minutes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {str(e)}")

@app.get("/leagues")
async def get_leagues():
    """Get available leagues"""
    return SAMPLE_LEAGUES

@app.get("/countries")
async def get_countries():
    """Get available countries"""
    return SAMPLE_COUNTRIES

@app.get("/markets")
async def get_markets():
    """Get available markets"""
    return list(MARKET_TYPES.keys())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

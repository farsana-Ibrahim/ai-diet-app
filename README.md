# 🍽 AI Diet Plan & Nutrition Recommendation API

An *AI-powered Diet Recommendation System* that analyzes user inputs, matches food intelligently using fuzzy NLP techniques, calculates nutrition, and generates a personalized 7-day diet plan.  
Built using *FastAPI, **Docker, and deployed on **Render Cloud*.

---

## 🚀 Live Demo

*Base URL:*  
https://ai-diet-app.onrender.com

*API Docs (Swagger UI):*  
https://ai-diet-app.onrender.com/docs

---

## 🧠 Features

### ✅ *AI / NLP-based Food Recognition*
- Uses *fuzzy string matching* (rapidfuzz + difflib)
- Understands misspelled food names  
  (e.g., "chapati", "chapathi", "chpati" → recognized as "chapathi")
- Maps food names to nutrition dataset

### ✅ *Nutrition Breakdown*
- Accepts a list of foods with quantities  
- Calculates calories, protein, carbs, and fat  
- Handles units like:  
  - grams (g), kilograms (kg)  
  - cups, tsp, tbsp  
  - pieces (roti, dosa, idli, banana, etc.)

### ✅ *Personalized 7-Day Diet Plan*
- Based on:
  - Age  
  - Weight  
  - Height  
  - Goal (weight loss / gain / muscle gain / maintenance)  
  - Cuisine preference (Vegetarian / Non-Vegetarian)  
  - Region (India-based foods)
- Generates balanced weekly meal plan  
- Uses calorie targets + food categories

### ✅ *Region-Based & Vegetarian Filtering*
- Filters Indian foods if region = “India”
- Removes all non-veg items if vegetarian preference is selected

### ✅ *Lightweight & Cloud Optimized*
- No heavy ML models  
- Fully optimized for Render’s 512MB free tier  
- Uses efficient pandas + fuzzy NLP for inference

---
### 1️⃣ *Health Check*
*GET* /health

{ "status": "healthy", "items_loaded": 120 }

---

### 2️⃣ *Nutrition Breakdown*
*POST* /nutrition-breakdown

#### Sample Input:
```json
{
  "foods": [
    {"item": "chapathi", "quantity": "2 pieces"},
    {"item": "banana", "quantity": "1 piece"},
    {"item": "rice", "quantity": "1 cup"}
  ]
}


---

3️⃣ Diet Plan Generation

POST /diet-plan

Sample Input:

{
  "name": "Farsana",
  "age": 23,
  "goal": "weight loss",
  "height": 160,
  "current_weight": 65,
  "cuisine_preference": "Vegetarian",
  "region": "India"
}

Returns a full 7-day diet plan with calorie-adjusted meals.


---

🧪 Testing the API (Swagger UI)

Visit:

➡ https://ai-diet-app.onrender.com/docs
You will see forms to test each endpoint.

---


☁ Deployment (Render)

Uses Docker Deployment
Deployed at: https://ai-diet-app.onrender.com

---

🛠 Tech Stack

Python 3.11
FastAPI
Uvicorn
Pandas
RapidFuzz (for fuzzy NLP)
Docker
Render Cloud

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import re
import os
from difflib import get_close_matches

# use rapidfuzz if available (faster fuzzy matching), otherwise fallback to difflib
try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None

app = FastAPI(title="AI Diet Plan & Nutrition API (Lightweight)")

# ---------------- LOAD DATA ----------------
CSV_PATHS = ["food_nutrition.csv", "/mnt/data/food_nutrition.csv"]
FOOD_DF = None
for p in CSV_PATHS:
    if os.path.exists(p):
        try:
            FOOD_DF = pd.read_csv(p)
            csv_path = p
            break
        except Exception as e:
            print("Failed to read CSV:", e)
            FOOD_DF = None

if FOOD_DF is None:
    FOOD_DF = pd.DataFrame(columns=["food", "region", "category",
                                    "calories_per_100g", "protein_g", "carbs_g", "fat_g"])
    FOOD_INDEX = []
    print("⚠ food_nutrition.csv not found. Using empty dataframe.")
else:
    FOOD_DF.columns = [c.strip() for c in FOOD_DF.columns]
    # Ensure column types exist
    for col in ["food", "region", "category", "calories_per_100g", "protein_g", "carbs_g", "fat_g"]:
        if col not in FOOD_DF.columns:
            FOOD_DF[col] = ""
    FOOD_INDEX = FOOD_DF["food"].astype(str).str.lower().tolist()
    print(f"✅ Loaded {len(FOOD_DF)} items from {csv_path}")



def parse_quantity(q: str):
    if not q:
        return None, "unknown"

    q = str(q).lower().strip()
    conversions = {
        "g": 1, "gm": 1, "gram": 1, "grams": 1,
        "kg": 1000,
        "ml": 1, "l": 1000,
        "tsp": 5, "tbsp": 15,
        "cup": 240
    }

    piece_weights = {
        "chapathi": 45, "roti": 45, "idli": 50,
        "dosa": 80, "egg": 50, "banana": 120,
        "apple": 150
    }

    m = re.match(r"([\d\.]+)\s*(\w+)", q)
    if m:
        num = float(m.group(1))
        unit = m.group(2)

        if unit in conversions:
            return num * conversions[unit], f"{num} {unit}"

        if unit in ["piece", "pieces", "pcs"]:
            base = 45
            for f, w in piece_weights.items():
                if f in q:
                    base = w
                    break
            return num * base, f"{int(num)} pieces"

    m2 = re.search(r"([\d\.]+)", q)
    if m2:
        return float(m2.group(1)), q

    return None, q


def scale_nutrients(row, grams: float):
    # numeric conversion with defaults
    try:
        cal = float(row.get("calories_per_100g", 0) or 0)
        protein = float(row.get("protein_g", 0) or 0)
        carbs = float(row.get("carbs_g", 0) or 0)
        fat = float(row.get("fat_g", 0) or 0)
    except Exception:
        cal = protein = carbs = fat = 0.0
    factor = grams / 100 if grams is not None else 0
    return {
        "calories": round(cal * factor, 2),
        "protein": round(protein * factor, 2),
        "carbs": round(carbs * factor, 2),
        "fat": round(fat * factor, 2),
    }


def semantic_food_match(q: str):
    return None


def fuzzy_food_match(q: str, threshold: int = 60):
    """
    Lightweight fuzzy: try exact -> contains -> rapidfuzz -> difflib
    """
    if FOOD_DF.empty:
        return None

    q = str(q).strip().lower()
    # exact match
    exact = FOOD_DF[FOOD_DF["food"].astype(str).str.lower() == q]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    # contains
    contains = FOOD_DF[FOOD_DF["food"].astype(str).str.lower().str.contains(q, na=False)]
    if not contains.empty:
        return contains.iloc[0].to_dict()

    # rapidfuzz (if installed)
    if process is not None and fuzz is not None and FOOD_INDEX:
        try:
            best = process.extractOne(q, FOOD_INDEX, scorer=fuzz.token_sort_ratio)
            if best:
                best_match, score, _ = best
                if score >= threshold:
                    matched = FOOD_DF[FOOD_DF["food"].astype(str).str.lower() == best_match].iloc[0].to_dict()
                    matched["confidence"] = score
                    return matched
        except Exception:
            pass

    # difflib fallback
    matches = get_close_matches(q, FOOD_INDEX, n=1, cutoff=0.5)
    if matches:
        return FOOD_DF[FOOD_DF["food"].astype(str).str.lower() == matches[0]].iloc[0].to_dict()

    return None


def find_food_match(item: str):
    s = semantic_food_match(item)
    if s:
        return s
    return fuzzy_food_match(item)


# ---------------- REQUEST MODELS ----------------
class FoodItem(BaseModel):
    item: str
    quantity: str

class NutritionRequest(BaseModel):
    foods: List[FoodItem]

class DietRequest(BaseModel):
    name: str
    age: int
    goal: str
    height: float
    current_weight: float
    target_weight: Optional[float] = None
    health_conditions: Optional[List[str]] = []
    cuisine_preference: Optional[str] = "Vegetarian"
    region: Optional[str] = "India"
    allergies: Optional[List[str]] = []


# ---------------- NUTRITION BREAKDOWN ----------------
@app.post("/nutrition-breakdown")
def nutrition_breakdown(req: NutritionRequest):
    if FOOD_DF.empty:
        raise HTTPException(status_code=500, detail="Nutrition dataset not loaded")

    output = []
    total_cals = total_pro = total_car = total_fat = 0.0

    for f in req.foods:
        grams, label = parse_quantity(f.quantity)
        if grams is None:
            raise HTTPException(status_code=400, detail=f"Invalid quantity: {f.quantity}")

        match = find_food_match(f.item)
        if match is None:
            output.append({
                "item": f"{f.item} ({label})",
                "calories": 0, "protein": 0, "carbs": 0, "fat": 0,
                "note": "Not found",
                "suggestions": []  
            })
            continue

        scaled = scale_nutrients(match, grams)
        output.append({
            "item": match.get("food", f.item),
            "calories": scaled["calories"],
            "protein": scaled["protein"],
            "carbs": scaled["carbs"],
            "fat": scaled["fat"]
        })

        total_cals += scaled["calories"]
        total_pro += scaled["protein"]
        total_car += scaled["carbs"]
        total_fat += scaled["fat"]

    return {
        "meal_nutrition": {
            "total_calories": round(total_cals, 2),
            "macros": {
                "protein": round(total_pro, 2),
                "carbs": round(total_car, 2),
                "fat": round(total_fat, 2)
            },
            "breakdown": output
        }
    }


# ---------------- 7-DAY BALANCED DIET PLAN ----------------
@app.post("/diet-plan")
def diet_plan(req: DietRequest):
    if FOOD_DF.empty:
        raise HTTPException(status_code=500, detail="Nutrition dataset not loaded")

    weight = req.current_weight
    height = req.height
    age = req.age

    # BMR (Mifflin-St Jeor)
    bmr = 10 * weight + 6.25 * height - 5 * age + 5
    tdee = bmr * 1.2

    goal = req.goal.lower()
    if "loss" in goal:
        target = tdee - 400
    elif "muscle" in goal:
        target = tdee + 300
    else:
        target = tdee

    target = max(1200, target)

    # ------------ REGION-BASED FILTERING ------------
    region = (req.region or "").lower()

    regional_df = FOOD_DF.copy()
    if "india" in region:
        try:
            regional_df = regional_df[regional_df["region"].astype(str).str.contains("India", case=False, na=False)]
        except Exception:
            regional_df = FOOD_DF.copy()

    if regional_df.empty:
        regional_df = FOOD_DF.copy()

    # ------------ CUISINE FILTER (VEGETARIAN / NON-VEG ) ------------
    cuisine = (req.cuisine_preference or "").lower()
    veg_blocklist = ["chicken", "fish", "mutton", "egg", "pork", "beef", "sausage", "seafood"]

    if "vegetarian" in cuisine:
        for item in veg_blocklist:
            regional_df = regional_df[~regional_df["food"].astype(str).str.lower().str.contains(item, na=False)]

    if regional_df.empty:
        regional_df = FOOD_DF.copy()

    # ------------ PICK FUNCTION (uses regional_df) ------------
    def pick(cat, n=2):
        df = regional_df[regional_df["category"].astype(str).str.contains(cat, case=False, na=False)]
        if df.empty:
            df = regional_df
        # If df.sample fails when len(df) == 0, fallback
        count = min(len(df), n) if len(df) > 0 else 0
        if count == 0:
            return []
        return df.sample(count).to_dict(orient="records")

    # ------------ BUILD WEEKLY DIET PLAN ------------
    week = {}
    for d in range(1, 8):
        breakfast = pick("Breakfast|Fruit|Dairy", 2)
        lunch = pick("Curry|Vegetable|Protein|Grain", 3)
        dinner = pick("Vegetable|Protein|Grain|Curry", 2)
        snacks = pick("Fruit|Nuts|Dairy|Vegetable", 1)

        def assemble(items, cal_target):
            out = []
            total = 0.0
            for it in items:
                scaled = scale_nutrients(it, 120)
                total += scaled["calories"]
                out.append({"food": it.get("food", ""), "serving_g": 120, "calories": scaled["calories"]})
            if total > 0:
                factor = cal_target / total
                for o in out:
                    o["calories"] = round(o["calories"] * factor, 2)
            return out

        week[f"day_{d}"] = {
            "target": round(target, 2),
            "meals": {
                "breakfast": assemble(breakfast, target * 0.25),
                "lunch": assemble(lunch, target * 0.35),
                "dinner": assemble(dinner, target * 0.30),
                "snacks": assemble(snacks, target * 0.10)
            }
        }

    return {
        "name": req.name,
        "goal": req.goal,
        "target_calories": round(target, 2),
        "diet_plan": week
    }


@app.get("/health")
def health():
    return {"status": "healthy", "items_loaded": len(FOOD_DF)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

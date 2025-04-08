# from ultralytics import YOLO
# import requests

# # Load YOLO model
# model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model

# # Detect food items in an image
# def detect_food(image_path):
#     results = model(image_path)
#     detected_foods = []
#     for result in results:
#         for obj in result.boxes:
#             food_name = model.names[int(obj.cls)]
#             detected_foods.append(food_name)
#     return list(set(detected_foods))

# # Get nutrition data from USDA API
# def get_usda_nutrition(food_name, api_key):
#     search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
#     params = {
#         "query": food_name,
#         "pageSize": 1,
#         "api_key": api_key
#     }
#     response = requests.get(search_url, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         if data.get("foods"):
#             # Extract FDC ID for the first result
#             fdc_id = data["foods"][0]["fdcId"]
            
#             # Get detailed nutrition info
#             details_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
#             details_response = requests.get(details_url, params={"api_key": api_key})
#             if details_response.status_code == 200:
#                 return details_response.json()
#     return {"error": "Nutrition data not found"}

# # Main function
# def main():
#     # Replace with your USDA API key
#     usda_api_key = "5GDlrAY1lKwsTvYqrR50c9jMRw6cXpgstyz2Ujp3"
#     image_path = "E:/DJANGO/HealthDisease/food_image.jpg"  # Replace with your image path

#     # Step 1: Detect food items
#     detected_foods = detect_food(image_path)
#     print(f"Detected food items: {detected_foods}")

#     # Step 2: Get nutrition details
#     total_nutrition = {
#         "calories": 0,
#         "protein": 0,
#         "fat": 0,
#         "carbs": 0,
#         "fiber": 0,
#         "sugars": 0,
#         "sodium": 0,
#         "iron": 0,
#         "calcium": 0,
#         "potassium": 0,
#         "vitamin_a": 0,
#         "vitamin_c": 0,
#     }
#     for food in detected_foods:
#         nutrition_data = get_usda_nutrition(food, usda_api_key)
#         #print(nutrition_data)
#         if "foodNutrients" in nutrition_data:
#             for nutrient in nutrition_data["foodNutrients"]:
#                 nutrient_name = nutrient["nutrient"].get("name", "")
#                 amount = nutrient.get("amount", 0)
                
#                 if nutrient_name == "Energy":
#                     total_nutrition["calories"] += amount
#                 elif nutrient_name == "Protein":
#                     total_nutrition["protein"] += amount
#                 elif nutrient_name == "Total lipid (fat)":
#                     total_nutrition["fat"] += amount
#                 elif nutrient_name == "Carbohydrate, by difference":
#                     total_nutrition["carbs"] += amount
#                 elif nutrient_name == "Fiber, total dietary":
#                     total_nutrition["fiber"] += amount
#                 elif nutrient_name == "Total Sugars":
#                     total_nutrition["sugars"] += amount
#                 elif nutrient_name == "Sodium, Na":
#                     total_nutrition["sodium"] += amount
#                 elif nutrient_name == "Iron, Fe":
#                     total_nutrition["iron"] += amount
#                 elif nutrient_name == "Calcium, Ca":
#                     total_nutrition["calcium"] += amount
#                 elif nutrient_name == "Potassium, K":
#                     total_nutrition["potassium"] += amount
#                 elif nutrient_name == "Vitamin A, IU":
#                     total_nutrition["vitamin_a"] += amount
#                 elif nutrient_name == "Vitamin C, total ascorbic acid":
#                     total_nutrition["vitamin_c"] += amount

#     # Step 3: Display total nutrition
#     print("Total Nutrition Details:")
#     print(total_nutrition)

# if __name__ == "__main__":
#     main()


import requests
from ultralytics import YOLO
from django.conf import settings

model = YOLO('yolov8n.pt')  

def detect_food(image_path):
    results = model(image_path)
    detected_foods = []
    for result in results:
        for obj in result.boxes:
            food_name = model.names[int(obj.cls)]
            detected_foods.append(food_name)
    return list(set(detected_foods))

def get_usda_nutrition(food_name, api_key):
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "pageSize": 1,
        "api_key": api_key
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            fdc_id = data["foods"][0]["fdcId"]
            details_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
            details_response = requests.get(details_url, params={"api_key": api_key})
            if details_response.status_code == 200:
                #print(details_response.json())
                return details_response.json()
    return {"error": "Nutrition data not found"}
 
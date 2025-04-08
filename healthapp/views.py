from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
import datetime

from sklearn.ensemble import GradientBoostingClassifier

from .forms import DoctorForm
from .models import *
from django.contrib.auth import authenticate, login, logout
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.



from pickle import encode_long
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from django.http import JsonResponse
from ultralytics import YOLO
import requests
from django.core.files.storage import FileSystemStorage
from .nutrition_helper import detect_food, get_usda_nutrition

# model = YOLO('yolov8n.pt')

# def detect_food(image_path):
#     results = model(image_path)
#     detected_foods = []
#     for result in results:
#         for obj in result.boxes:
#             food_name = model.names[int(obj.cls)]
#             detected_foods.append(food_name)
#     return list(set(detected_foods))

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
#             fdc_id = data["foods"][0]["fdcId"]
#             details_url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
#             details_response = requests.get(details_url, params={"api_key": api_key})
#             if details_response.status_code == 200:
#                 return details_response.json()
#     return {"error": "Nutrition data not found"}

# def upload_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         image = request.FILES['image']
#         image_path = 'media/' + image.name
#         with open(image_path, 'wb') as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         usda_api_key = "5GDlrAY1lKwsTvYqrR50c9jMRw6cXpgstyz2Ujp3"
#         detected_foods = detect_food(image_path)

#         total_nutrition = {
#             "calories": 0,
#             "protein": 0,
#             "fat": 0,
#             "carbs": 0,
#             "fiber": 0,
#             "sugars": 0,
#             "sodium": 0,
#             "iron": 0,
#             "calcium": 0,
#             "potassium": 0,
#             "vitamin_a": 0,
#             "vitamin_c": 0,
#         }

#         for food in detected_foods:
#             nutrition_data = get_usda_nutrition(food, usda_api_key)
#             if "foodNutrients" in nutrition_data:
#                 for nutrient in nutrition_data["foodNutrients"]:
#                     nutrient_name = nutrient["nutrient"].get("name", "")
#                     amount = nutrient.get("amount", 0)
#                     if nutrient_name == "Energy":
#                         total_nutrition["calories"] += amount
#                     elif nutrient_name == "Protein":
#                         total_nutrition["protein"] += amount
#                     elif nutrient_name == "Total lipid (fat)":
#                         total_nutrition["fat"] += amount
#                     elif nutrient_name == "Carbohydrate, by difference":
#                         total_nutrition["carbs"] += amount
#                     elif nutrient_name == "Fiber, total dietary":
#                         total_nutrition["fiber"] += amount
#                     elif nutrient_name == "Total Sugars":
#                         total_nutrition["sugars"] += amount
#                     elif nutrient_name == "Sodium, Na":
#                         total_nutrition["sodium"] += amount
#                     elif nutrient_name == "Iron, Fe":
#                         total_nutrition["iron"] += amount
#                     elif nutrient_name == "Calcium, Ca":
#                         total_nutrition["calcium"] += amount
#                     elif nutrient_name == "Potassium, K":
#                         total_nutrition["potassium"] += amount
#                     elif nutrient_name == "Vitamin A, IU":
#                         total_nutrition["vitamin_a"] += amount
#                     elif nutrient_name == "Vitamin C, total ascorbic acid":
#                         total_nutrition["vitamin_c"] += amount

#         return render(request, 'result.html', {'nutrition_data': total_nutrition})

#     return render(request, 'upload.html')



# # Render the upload image page
# def upload_image(request):
#     if request.method == 'POST':
#         # Save the uploaded image
#         uploaded_file = request.FILES['image']
#         fs = FileSystemStorage()
#         file_path = fs.save(uploaded_file.name, uploaded_file)
#         full_path = fs.path(file_path)

#         # Detect food items using YOLO
#         detected_foods = detect_food(full_path)

#         # Fetch nutrition data
#         usda_api_key = "5GDlrAY1lKwsTvYqrR50c9jMRw6cXpgstyz2Ujp3"
#         total_nutrition = {
#             "calories": 0, "protein": 0, "fat": 0, "carbs": 0,
#             "fiber": 0, "sugars": 0, "sodium": 0, "iron": 0,
#             "calcium": 0, "potassium": 0, "vitamin_a": 0, "vitamin_c": 0,
#         }

#         for food in detected_foods:
#             nutrition_data = get_usda_nutrition(food, usda_api_key)
#             if "foodNutrients" in nutrition_data:
#                 for nutrient in nutrition_data["foodNutrients"]:
#                     nutrient_name = nutrient["nutrient"].get("name", "")
#                     amount = nutrient.get("amount", 0)

#                     if nutrient_name == "Energy":
#                         total_nutrition["calories"] += amount
#                     elif nutrient_name == "Protein":
#                         total_nutrition["protein"] += amount
#                     elif nutrient_name == "Total lipid (fat)":
#                         total_nutrition["fat"] += amount
#                     elif nutrient_name == "Carbohydrate, by difference":
#                         total_nutrition["carbs"] += amount
#                     elif nutrient_name == "Fiber, total dietary":
#                         total_nutrition["fiber"] += amount
#                     elif nutrient_name == "Total Sugars":
#                         total_nutrition["sugars"] += amount
#                     elif nutrient_name == "Sodium, Na":
#                         total_nutrition["sodium"] += amount
#                     elif nutrient_name == "Iron, Fe":
#                         total_nutrition["iron"] += amount
#                     elif nutrient_name == "Calcium, Ca":
#                         total_nutrition["calcium"] += amount
#                     elif nutrient_name == "Potassium, K":
#                         total_nutrition["potassium"] += amount
#                     elif nutrient_name == "Vitamin A, IU":
#                         total_nutrition["vitamin_a"] += amount
#                     elif nutrient_name == "Vitamin C, total ascorbic acid":
#                         total_nutrition["vitamin_c"] += amount

#         # Store the results in the session and redirect to results page
#         request.session['detected_foods'] = detected_foods
#         request.session['total_nutrition'] = total_nutrition
#         return redirect('image_result')

#     return render(request, 'upload_image.html')


# # Render the results page
# def image_result(request):
#     detected_foods = request.session.get('detected_foods', [])
#     total_nutrition = request.session.get('total_nutrition', {})
#     return render(request, 'image_result.html', {
#         'detected_foods': detected_foods,
#         'total_nutrition': total_nutrition,
#     })

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO
import requests
import os


diet_plan_df = pd.read_csv("Diet_Plan.csv")

def get_diet_plan(disease):
    """Fetch diet plan based on the predicted disease from CSV."""
    diet_info = diet_plan_df[diet_plan_df["Disease"].str.lower() == disease.lower()]
    if not diet_info.empty:
        full_plan = diet_info.iloc[0]["Diet Plan"]
        
        # Split the diet plan into Allowed Foods and Avoid Foods
        if "Avoid Foods:" in full_plan:
            parts = full_plan.split("Avoid Foods:")
            allowed_foods = parts[0].replace("Allowed Foods:", "").strip().split("- ")
            avoid_foods = parts[1].strip().split("- ")
        else:
            allowed_foods = full_plan.replace("Allowed Foods:", "").strip().split("- ")
            avoid_foods = []

        return {"allowed_foods": [item.strip() for item in allowed_foods if item], 
                "avoid_foods": [item.strip() for item in avoid_foods if item]}
    
    return {"allowed_foods": [], "avoid_foods": ["No specific diet plan available."]}


def nutrition_details(request):
    if request.method == 'POST' and request.FILES['food_image']:
        uploaded_file = request.FILES['food_image']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Detect food items
        detected_foods = detect_food(full_file_path)

        # Get nutrition details
        usda_api_key = "5GDlrAY1lKwsTvYqrR50c9jMRw6cXpgstyz2Ujp3"
        nutrition_details = {}
        for food in detected_foods:
            nutrition_data = get_usda_nutrition(food, usda_api_key)
            if "foodNutrients" in nutrition_data:
                food_nutrients = {}
                for nutrient in nutrition_data["foodNutrients"]:
                    nutrient_name = nutrient["nutrient"].get("name", "")
                    amount = nutrient.get("amount", 0)
                    food_nutrients[nutrient_name] = amount
                nutrition_details[food] = food_nutrients

        # Delete uploaded file
        fs.delete(file_path)

        return render(request, 'nutrition_details.html', {
            'detected_foods': detected_foods,
            'nutrition_details': nutrition_details
        })

    return render(request, 'nutrition_details.html')


def Home(request):
    return render(request,'carousel.html')

def Admin_Home(request):
    dis = Search_Data.objects.all()
    pat = Patient.objects.all()
    doc = Doctor.objects.all()
    feed = Feedback.objects.all()

    d = {'dis':dis.count(),'pat':pat.count(),'doc':doc.count(),'feed':feed.count()}
    return render(request,'admin_home.html',d)

@login_required(login_url="login")
def assign_status(request,pid):
    doctor = Doctor.objects.get(id=pid)
    if doctor.status == 1:
        doctor.status = 2
        messages.success(request, 'Selected doctor are successfully withdraw his approval.')
    else:
        doctor.status = 1
        messages.success(request, 'Selected doctor are successfully approved.')
    doctor.save()
    return redirect('view_doctor')

@login_required(login_url="login")
def User_Home(request):
    return render(request,'patient_home.html')

@login_required(login_url="login")
def Doctor_Home(request):
    return render(request,'doctor_home.html')

def About(request):
    return render(request,'about.html')

def Contact(request):
    return render(request,'contact.html')


def Gallery(request):
    return render(request,'gallery.html')


def Login_User(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        sign = ""
        if user:
            try:
                sign = Patient.objects.get(user=user)
            except:
                pass
            if sign:
                login(request, user)
                error = "pat1"
            else:
                pure=False
                try:
                    pure = Doctor.objects.get(status=1,user=user)
                except:
                    pass
                if pure:
                    login(request, user)
                    error = "pat2"
                else:
                    login(request, user)
                    error="notmember"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'login.html', d)

def Login_admin(request):
    error = ""
    if request.method == "POST":
        u = request.POST['uname']
        p = request.POST['pwd']
        user = authenticate(username=u, password=p)
        if user.is_staff:
            login(request, user)
            error="pat"
        else:
            error="not"
    d = {'error': error}
    return render(request, 'admin_login.html', d)

def Signup_User(request):
    error = ""
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        u = request.POST['uname']
        e = request.POST['email']
        p = request.POST['pwd']
        d = request.POST['dob']
        con = request.POST['contact']
        add = request.POST['add']
        type = request.POST['type']
        im = request.FILES['image']
        dat = datetime.date.today()
        user = User.objects.create_user(email=e, username=u, password=p, first_name=f,last_name=l)
        if type == "Patient":
            Patient.objects.create(user=user,contact=con,address=add,image=im,dob=d)
        else:
            Doctor.objects.create(dob=d,image=im,user=user,contact=con,address=add,status=2)
        error = "create"
    d = {'error':error}
    return render(request,'register.html',d)

def Logout(request):
    logout(request)
    return redirect('home')

@login_required(login_url="login")
def Change_Password(request):
    sign = 0
    user = User.objects.get(username=request.user.username)
    error = ""
    if not request.user.is_staff:
        try:
            sign = Patient.objects.get(user=user)
            if sign:
                error = "pat"
        except:
            sign = Doctor.objects.get(user=user)
    terror = ""
    if request.method=="POST":
        n = request.POST['pwd1']
        c = request.POST['pwd2']
        o = request.POST['pwd3']
        if c == n:
            u = User.objects.get(username__exact=request.user.username)
            u.set_password(n)
            u.save()
            terror = "yes"
        else:
            terror = "not"
    d = {'error':error,'terror':terror,'data':sign}
    return render(request,'change_password.html',d)


def preprocess_inputs(df, scaler):
    df = df.copy()
    # Split df into X and y
    y = df['target'].copy()
    X = df.drop('target', axis=1).copy()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y


def prdict_heart_disease(list_data):
    csv_file = Admin_Helath_CSV.objects.get(id=1)
    df = pd.read_csv(csv_file.csv_file)

    X = df[['age','sex','cp',  'trestbps',  'chol',  'fbs',  'restecg',  'thalach',  'exang',  'oldpeak',  'slope',  'ca',  'thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
    nn_model = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
    nn_model.fit(X_train, y_train)
    pred = nn_model.predict([list_data])
    print("Neural Network Accuracy: {:.2f}%".format(nn_model.score(X_test, y_test) * 100))
    print("Prdicted Value is : ", format(pred))
    dataframe = str(df.head())
    return (nn_model.score(X_test, y_test) * 100),(pred)

@login_required(login_url="login")
def add_doctor(request,pid=None):
    doctor = None
    if pid:
        doctor = Doctor.objects.get(id=pid)
    if request.method == "POST":
        form = DoctorForm(request.POST, request.FILES, instance = doctor)
        if form.is_valid():
            new_doc = form.save()
            new_doc.status = 1
            if not pid:
                user = User.objects.create_user(password=request.POST['password'], username=request.POST['username'], first_name=request.POST['first_name'], last_name=request.POST['last_name'])
                new_doc.user = user
            new_doc.save()
            return redirect('view_doctor')
    d = {"doctor": doctor}
    return render(request, 'add_doctor.html', d)

@login_required(login_url="login")
def add_heartdetail(request):
    if request.method == "POST":
        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        list_data = []
        value_dict = eval(str(request.POST)[12:-1])
        count = 0
        for key,value in value_dict.items():
            if count == 0:
                count =1
                continue
            if key == "sex" and value[0] == "Male" or value[0] == 'male' or value[0]=='m' or value[0] == 'M':
                list_data.append(0)
                continue
            elif key == "sex":
                list_data.append(1)
                continue
            list_data.append(value[0])

        # list_data = [57, 0, 1, 130, 236, 0, 0, 174, 0, 0.0, 1, 1, 2]
        accuracy,pred = prdict_heart_disease(list_data)
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=round(accuracy,2), result=pred[0], values_list=list_data,predict_for="Heart Prediction")
        rem = int(pred[0])
        print("Result = ",rem)
        if pred[0] == 0:
            pred = "<span style='color:green'>You are healthy</span>"
        else:
            pred = "<span style='color:red'>You are Unhealthy, Need to Checkup.</span>"
        return redirect('predict_disease', str(rem), str(round(accuracy,2)))
    return render(request, 'add_heartdetail.html')

@login_required(login_url="login")
def predict_disease(request, pred, accuracy):
    doctor = Doctor.objects.filter(address__icontains=Patient.objects.get(user=request.user).address)
    d = {'pred': pred, 'accuracy':accuracy, 'doctor':doctor}
    return render(request, 'predict_disease.html', d)

@login_required(login_url="login")
def view_search_pat(request):
    doc = None
    try:
        doc = Doctor.objects.get(user=request.user)
        data = Search_Data.objects.filter(patient__address__icontains=doc.address).order_by('-id')
    except:
        try:
            doc = Patient.objects.get(user=request.user)
            data = Search_Data.objects.filter(patient=doc).order_by('-id')
        except:
            data = Search_Data.objects.all().order_by('-id')
    return render(request,'view_search_pat.html',{'data':data})

@login_required(login_url="login")
def delete_doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    doc.delete()
    return redirect('view_doctor')

@login_required(login_url="login")
def delete_feedback(request,pid):
    doc = Feedback.objects.get(id=pid)
    doc.delete()
    return redirect('view_feedback')

@login_required(login_url="login")
def delete_patient(request,pid):
    doc = Patient.objects.get(id=pid)
    doc.delete()
    return redirect('view_patient')

@login_required(login_url="login")
def delete_searched(request,pid):
    doc = Search_Data.objects.get(id=pid)
    doc.delete()
    return redirect('view_search_pat')

@login_required(login_url="login")
def View_Doctor(request):
    doc = Doctor.objects.all()
    d = {'doc':doc}
    return render(request,'view_doctor.html',d)

@login_required(login_url="login")
def View_Patient(request):
    patient = Patient.objects.all()
    d = {'patient':patient}
    return render(request,'view_patient.html',d)

@login_required(login_url="login")
def View_Feedback(request):
    dis = Feedback.objects.all()
    d = {'dis':dis}
    return render(request,'view_feedback.html',d)

@login_required(login_url="login")
def View_My_Detail(request):
    terror = ""
    user = User.objects.get(id=request.user.id)
    error = ""
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    d = {'error': error,'pro':sign}
    return render(request,'profile_doctor.html',d)

@login_required(login_url="login")
def Edit_Doctor(request,pid):
    doc = Doctor.objects.get(id=pid)
    error = ""
    # type = Type.objects.all()
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        cat = request.POST['type']
        try:
            im = request.FILES['image']
            doc.image=im
            doc.save()
        except:
            pass
        dat = datetime.date.today()
        doc.user.first_name = f
        doc.user.last_name = l
        doc.user.email = e
        doc.contact = con
        doc.category = cat
        doc.address = add
        doc.user.save()
        doc.save()
        error = "create"
    d = {'error':error,'doc':doc,'type':type}
    return render(request,'edit_doctor.html',d)

@login_required(login_url="login")
def Edit_My_deatail(request):
    terror = ""
    print("Hii welcome")
    user = User.objects.get(id=request.user.id)
    error = ""
    # type = Type.objects.all()
    try:
        sign = Patient.objects.get(user=user)
        error = "pat"
    except:
        sign = Doctor.objects.get(user=user)
    if request.method == 'POST':
        f = request.POST['fname']
        l = request.POST['lname']
        e = request.POST['email']
        con = request.POST['contact']
        add = request.POST['add']
        try:
            im = request.FILES['image']
            sign.image = im
            sign.save()
        except:
            pass
        to1 = datetime.date.today()
        sign.user.first_name = f
        sign.user.last_name = l
        sign.user.email = e
        sign.contact = con
        if error != "pat":
            cat = request.POST['type']
            sign.category = cat
            sign.save()
        sign.address = add
        sign.user.save()
        sign.save()
        terror = "create"
    d = {'error':error,'terror':terror,'doc':sign}
    return render(request,'edit_profile.html',d)

@login_required(login_url='login')
def sent_feedback(request):
    terror = None
    if request.method == "POST":
        username = request.POST['uname']
        message = request.POST['msg']
        username = User.objects.get(username=username)
        Feedback.objects.create(user=username, messages=message)
        terror = "create"
    return render(request, 'sent_feedback.html',{'terror':terror})


def add_generalhealth(request):
    predictiondata = None
    diet_plan = None
    deseaseli = []
    if request.method=="POST":
        # for i,j in request.POST.items():
        #     if "csrfmiddlewaretoken" != i:
        #         deseaseli.append(i)
        deseaseli = request.POST.getlist('selected_symptoms')
        if not deseaseli:
            print("No symptoms selected.")
            return render(request, 'add_genralhealth.html', {'alldisease': alldisease, 'predictiondata': None})
        
        # Process the symptoms as needed
        print(f"Selected symptoms: {deseaseli}")

        # training.csv
        DATA_PATH = Admin_Helath_CSV.objects.get(id=2)
        #df = pd.read_csv(csv_file.csv_file)
        data = pd.read_csv(DATA_PATH.csv_file).dropna(axis = 1)

        # Checking whether the dataset is balanced or not
        disease_counts = data["prognosis"].value_counts()
        temp_df = pd.DataFrame({
            "Disease": disease_counts.index,
            "Counts": disease_counts.values
        })

        plt.figure(figsize = (18,8))
        sns.barplot(x = "Disease", y = "Counts", data = temp_df)
        plt.xticks(rotation=90)
        # plt.show()

        # Encoding the target value into numerical
        # value using LabelEncoder
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])


        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size = 0.2, random_state = 24)



        # symptoms = X.columns.values
        # symptom_index = {}
        # for index, value in enumerate(symptoms):
        #     symptom = " ".join([i.capitalize() for i in value.split("_")])
        #     symptom_index[symptom] = index
        
        symptoms = X.columns.values
        symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }


        data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }

        final_svm_model = SVC()
        final_nb_model = GaussianNB()
        final_rf_model = RandomForestClassifier(random_state=18)
        final_svm_model.fit(X, y)
        final_nb_model.fit(X, y)
        final_rf_model.fit(X, y)

        #Testing.csv
        DATA_PATH2 = Admin_Helath_CSV.objects.get(id=3)
        test_data = pd.read_csv(DATA_PATH2.csv_file).dropna(axis=1)

        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])

        svm_preds = final_svm_model.predict(test_X)
        nb_preds = final_nb_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)

        print("svm_preds:", svm_preds)
        print("nb_preds:", nb_preds)
        print("rf_preds:", rf_preds)
        print("Length of svm_preds:", len(svm_preds))
        print("Length of nb_preds:", len(nb_preds))
        print("Length of rf_preds:", len(rf_preds))


        
        final_preds = []
        for i, j, k in zip(svm_preds, nb_preds, rf_preds):
            predictions = [i, j, k]
            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            final_preds.append(most_common_prediction)
            
        print(f"Accuracy on Test dataset by the combined model\
        : {accuracy_score(test_Y, final_preds)*100}")

        cf_matrix = confusion_matrix(test_Y, final_preds)
        plt.figure(figsize=(12,8))

        sns.heatmap(cf_matrix, annot = True)
        

        def predictDisease(symptoms):
            input_data = [0] * len(data_dict["symptom_index"])
            for symptom in symptoms:
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1

            input_data = np.array(input_data).reshape(1, -1)

            rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
            nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
            svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
 
            predictions = [rf_prediction, nb_prediction, svm_prediction]
            most_common_prediction = Counter(predictions).most_common(1)[0][0]

            return {
                "RandomForestClassifier Prediction": rf_prediction,
                "GaussianNB Prediction": nb_prediction,
                "SVC Prediction": svm_prediction,
                "Final Prediction": most_common_prediction
            }

        # Testing the function
        predictiondata = predictDisease(deseaseli)
        
        diet_plan = get_diet_plan(predictiondata['Final Prediction'])
        
        patient = Patient.objects.get(user=request.user)
        Search_Data.objects.create(patient=patient, prediction_accuracy=round(accuracy_score(test_Y, final_preds)*100,2), result=predictiondata["Final Prediction"], values_list=deseaseli, predict_for="General Health Prediction")

        # print(deseaseli)
    alldisease = ['Itching','Skin Rash','Nodal Skin Eruptions','Continuous Sneezing','Shivering','Chills','Joint Pain',	'Stomach Pain','Acidity','Ulcers On Tongue','Muscle Wasting','Vomiting','Burning Micturition','Fatigue','Weight Gain','Anxiety','Cold Hands And Feets','Mood Swings','Weight Loss','Restlessness','Lethargy','Patches In Throat','Irregular Sugar Level','Cough','High Fever','Sunken Eyes','Breathlessness','Sweating','Dehydration',	'Indigestion','Headache','Yellowish Skin','Dark Urine','Nausea','Loss Of Appetite','Pain Behind The Eyes','Back Pain','Constipation','Abdominal Pain','Diarrhoea','Mild Fever','Yellow Urine','Yellowing Of Eyes','Acute Liver Failure','Fluid Overload','Swelling Of Stomach','Swelled Lymph Nodes','Malaise','Blurred And Distorted Vision','Phlegm','Throat Irritation','Redness Of Eyes','Sinus Pressure','Runny Nose','Congestion','Chest Pain','Weakness In Limbs','Fast Heart Rate',	'Pain During Bowel Movements','Pain In Anal Region','Bloody Stool','Irritation In Anus','Neck Pain','Dizziness','Cramps','Bruising','Obesity','Swollen Legs','Swollen Blood Vessels','Puffy Face And Eyes','Enlarged Thyroid','Brittle Nails','Swollen Extremeties','Excessive Hunger','Extra Marital Contacts','Drying And Tingling Lips','Slurred Speech','Knee Pain','Hip Joint Pain','Muscle Weakness','Stiff Neck','Swelling Joints','Movement Stiffness','Spinning Movements','Loss Of Balance','Unsteadiness','Weakness Of One Body Side','Loss Of Smell','Bladder Discomfort','Continuous Feel Of Urine','Passage Of Gases','Internal Itching',	'Depression','Irritability','Muscle Pain','Altered Sensorium','Red Spots Over Body','Belly Pain','Abnormal Menstruation','Dischromic Patches','Watering From Eyes','Increased Appetite','Polyuria','Family History','Mucoid Sputum','Rusty Sputum','Lack Of Concentration',	'Visual Disturbances','Receiving Blood Transfusion','Receiving Unsterile Injections','Coma','Stomach Bleeding',	'Distention Of Abdomen','History Of Alcohol Consumption','Fluid Overload','Blood In Sputum','Prominent Veins On Calf','Palpitations','Painful Walking','Pus Filled Pimples', 'Blackheads','Scurring','Skin Peeling','Silver Like Dusting','Small Dents In Nails','Inflammatory Nails','Blister','Red Sore Around Nose','Yellow Crust Ooze']
    return render(request,'add_genralhealth.html', {'alldisease':alldisease, 'predictiondata':predictiondata,'diet_plan':diet_plan})


def search_blood(request):
    data = Blood_Donation.objects.filter(status="Approved")
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Request for Blood", status="Pending", place=place)
        messages.success(request, "Request Generated.")
        return redirect('search_blood')
    return render(request, 'search_blood.html', {'data':data})


def donate_blood(request):
    if request.method == "POST":
        bg = request.POST['bg']
        place = request.POST['place']
        user = Patient.objects.get(user=request.user)
        data = Blood_Donation.objects.create(blood_group=bg, user=user, purpose="Blood Donor", status="Pending", place=place)
        messages.success(request, "Added Your Detail.")
        return redirect('donate_blood')
    return render(request, 'donate_blood.html')

def request_blood(request):
    mydata = request.GET.get('action',0)
    data = Blood_Donation.objects.filter(purpose="Request for Blood")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'request_blood.html', {'data':data})

def donator_blood(request):
    mydata = request.GET.get('action',0)
    data = Blood_Donation.objects.filter(purpose="Blood Donor")
    if mydata:
        data = data.filter(status=mydata)
    return render(request, 'donator_blood.html', {'data':data})

def change_status(request,pid):
    data = Blood_Donation.objects.get(id=pid)
    url = request.GET.get('data')
    if data.status == "Approved":
        data.status = "Pending"
        data.save()
    else:
        data.status = "Approved"
        data.save()
    return HttpResponseRedirect(url)
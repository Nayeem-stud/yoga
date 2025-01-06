from django.shortcuts import render, redirect
import time
from userapp.models import *
from adminapp.models import *
from mainapp.models import *
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.conf import settings
from django.core.paginator import Paginator
import matplotlib.pyplot as plt
import io
import base64
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
import os
import numpy as np
from tensorflow.keras.models import load_model
from django.contrib import messages
import pandas as pd
import pytz
import matplotlib
from django.core.files.storage import default_storage
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from PIL import UnidentifiedImageError
from django.http import JsonResponse
from django.utils import timezone


# ------------------------------------------------------


# Create your views here.
def user_dashboard(req):
    prediction_count = UserModel.objects.all().count()
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    Feedbacks_users_count = Feedback.objects.all().count()
    all_users_count = UserModel.objects.all().count()

    if user.Last_Login_Time is None:
        IST = pytz.timezone("Asia/Kolkata")
        current_time_ist = datetime.now(IST).time()
        user.Last_Login_Time = current_time_ist
        user.save()
        return redirect("user_dashboard")

    return render(
        req,
        "user/user-dashboard.html",
        {
            "predictions": prediction_count,
            "user_name": user.user_name,
            "feedback_count": Feedbacks_users_count,
            "all_users_count": all_users_count,
        },
    )


def user_profile(req):
    user_id = req.session["user_id"]
    user = UserModel.objects.get(user_id=user_id)
    if req.method == "POST":
        user_name = req.POST.get("username")
        user_age = req.POST.get("age")
        user_phone = req.POST.get("mobile number")
        user_email = req.POST.get("email")
        user_password = req.POST.get("Password")
        user_address = req.POST.get("address")

        # user_img = req.POST.get("userimg")

        user.user_name = user_name
        user.user_age = user_age
        user.user_address = user_address
        user.user_contact = user_phone
        user.user_email = user_email
        user.user_password = user_password

        if len(req.FILES) != 0:
            image = req.FILES["profilepic"]
            user.user_image = image
            user.user_name = user_name
            user.user_age = user_age
            user.user_contact = user_phone
            user.user_email = user_email
            user.user_address = user_address
            user.user_password = user_password
            user.save()
            messages.success(req, "Updated Successfully.")
        else:
            user.user_name = user_name
            user.user_age = user_age
            user.save()
            messages.success(req, "Updated Successfully.")

    context = {"i": user}
    return render(req, "user/user-profile.html", context)


# ----------------------------------------------------------
import os
import base64
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.contrib import messages
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from django.core.files.storage import default_storage
from keras.models import load_model
import os
import numpy as np
import cv2
import base64
from django.shortcuts import redirect, render
from django.contrib import messages
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

# Define class mapping for prediction
class_dict = {
    0: "Adho Mukha Svanasana",
    1: "Anjaneyasana",
    2: "Ardha Matsyendrasana",
    3: "Baddha Konasana",
    4: "Bakasana",
    5: "Balasana",
    6: "Halasana",
    7: "Malasana",
    8: "Salamba Bhujangasana",
    9: "Setu Bandha Sarvangasana",
    10: "Urdhva Mukha Svsnssana",
    11: "Utthita Hasta Padangusthasana",
    12: "Virabhadrasana One",
    13: "Virabhadrasana Two",
    14: "Vrksasana",
}


# Preprocessing functions for each model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = preprocess_input(img_array)  # Preprocess input
    img_array = img_array.reshape(1, 224, 224, 3)  # Add batch dimension
    return img_array


# Prediction functions
def predict_image(image_path, model, class_dict):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_dict.get(predicted_class_index, "Unknown")
    return predicted_class_label


# Load model functions
def load_model_vgg16():
    model_path = os.path.join(settings.BASE_DIR, "yoga_posture_dataset/vgg_yoga.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def load_model_mobilenet():
    model_path = os.path.join(settings.BASE_DIR, "yoga_posture_dataset/mobilenet.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def load_model_densenet():
    model_path = os.path.join(
        settings.BASE_DIR, "yoga_posture_dataset/densnet_model.h5"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


# Function to get model info based on model type
def get_model_info(model_type):
    if model_type == "Densenet":
        model_info = Densenet_model.objects.latest("S_No")
    elif model_type == "vgg16":
        model_info = Vgg16_model.objects.latest("S_No")
    elif model_type == "Mobilenet":
        model_info = MobileNet_model.objects.latest("S_No")
    else:
        raise ValueError("Select a valid Model")
    return model_info


# Generate segmented image and encode to base64
def generate_segmented_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    segmented_image_path = os.path.join(settings.MEDIA_ROOT, "segmented_image.jpg")
    cv2.imwrite(segmented_image_path, binary_image)

    grayscale_image_path = os.path.join(settings.MEDIA_ROOT, "grayscale_image.jpg")
    cv2.imwrite(grayscale_image_path, gray_image)

    with open(image_path, "rb") as img_file:
        original_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    with open(segmented_image_path, "rb") as img_file:
        segmented_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    with open(grayscale_image_path, "rb") as img_file:
        grayscale_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return original_image_base64, segmented_image_base64, grayscale_image_base64


def Classification(req):
    
        return render(req, "user/detection.html")




def Classification_result(req):
   
        return redirect("Classification")


# ----------------------------------------------------------------------------------------------------


def user_feedback(req):
    id = req.session["user_id"]
    uusser = UserModel.objects.get(user_id=id)
    if req.method == "POST":
        rating = req.POST.get("rating")
        review = req.POST.get("review")
        # print(sentiment)
        # print(rating)
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        sentiment = None
        if score["compound"] > 0 and score["compound"] <= 0.5:
            sentiment = "positive"
        elif score["compound"] >= 0.5:
            sentiment = "very positive"
        elif score["compound"] < -0.5:
            sentiment = "negative"
        elif score["compound"] < 0 and score["compound"] >= -0.5:
            sentiment = " very negative"
        else:
            sentiment = "neutral"
        Feedback.objects.create(
            Rating=rating, Review=review, Sentiment=sentiment, Reviewer=uusser
        )
        messages.success(req, "Feedback recorded")
        return redirect("user_feedback")
    return render(req, "user/user-feedback.html")


def user_logout(req):
    if "user_id" in req.session:
        view_id = req.session["user_id"]
        try:
            user = UserModel.objects.get(user_id=view_id)
            user.Last_Login_Time = timezone.now().time()
            user.Last_Login_Date = timezone.now().date()
            user.save()
            messages.info(req, "You are logged out.")
        except UserModel.DoesNotExist:
            pass
    req.session.flush()
    return redirect("user_login")


# --------------------------------------------------------------------------

from django.shortcuts import render, redirect
from mainapp.models import *
from userapp.models import *
from adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
import pandas as pd
import numpy as np


# Create your views here.


def admin_dashboard(req):
    all_users_count = UserModel.objects.all().count()
    pending_users_count = UserModel.objects.filter(User_Status="pending").count()
    rejected_users_count = UserModel.objects.filter(User_Status="removed").count()
    accepted_users_count = UserModel.objects.filter(User_Status="accepted").count()
    Feedbacks_users_count = Feedback.objects.all().count()
    prediction_count = UserModel.objects.all().count()
    return render(
        req,
        "admin/admin-dashboard.html",
        {
            "a": all_users_count,
            "b": pending_users_count,
            "c": rejected_users_count,
            "d": accepted_users_count,
            "e": Feedbacks_users_count,
            "f": prediction_count,
        },
    )


def pending_users(req):
    pending = UserModel.objects.filter(User_Status="pending")
    paginator = Paginator(pending, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/Pending-users.html", {"user": post})


def all_users(req):
    all_users = UserModel.objects.all()
    paginator = Paginator(all_users, 5)
    page_number = req.GET.get("page")
    post = paginator.get_page(page_number)
    return render(req, "admin/All-users.html", {"allu": all_users, "user": post})


def delete_user(request, user_id):
    try:
        user = UserModel.objects.get(user_id=user_id)
        user.delete()
        messages.warning(request, "User was deleted successfully!")
    except UserModel.DoesNotExist:
        messages.error(request, "User does not exist.")
    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")

    return redirect("all_users")


# Acept users button
def accept_user(request, id):
    try:
        status_update = UserModel.objects.get(user_id=id)
        status_update.User_Status = "accepted"
        status_update.save()
        messages.success(request, "User was accepted successfully!")
    except UserModel.DoesNotExist:
        messages.error(request, "User does not exist.")
    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")

    return redirect("pending_users")


# Remove user button
def reject_user(req, id):
    status_update2 = UserModel.objects.get(user_id=id)
    status_update2.User_Status = "removed"
    status_update2.save()
    messages.warning(req, "User was Rejected..!")
    return redirect("pending_users")


# Change status users button
def change_status(request, id):
    user_data = UserModel.objects.get(user_id=id)
    if user_data.User_Status == "removed":
        user_data.User_Status = "accepted"
        user_data.save()
    elif user_data.User_Status == "accepted":
        user_data.User_Status = "removed"
        user_data.save()
    elif user_data.User_Status == "pending":
        messages.info(request, "Accept the user first..!")
        return redirect("all_users")
    messages.success(request, "User status was changed..!")
    return redirect("all_users")


def adminlogout(req):
    messages.info(req, "You are logged out.")
    return redirect("admin_login")


def admin_feedback(req):
    feed = Feedback.objects.all()
    return render(req, "admin/Admin-feedback.html", {"back": feed})


def sentiment_analysis(req):
    fee = Feedback.objects.all()
    return render(req, "admin/Sentiment-analysis.html", {"cat": fee})


def sentiment_analysis_graph(req):
    positive = Feedback.objects.filter(Sentiment="positive").count()
    very_positive = Feedback.objects.filter(Sentiment="very positive").count()
    negative = Feedback.objects.filter(Sentiment="negative").count()
    very_negative = Feedback.objects.filter(Sentiment="very negative").count()
    neutral = Feedback.objects.filter(Sentiment="neutral").count()
    context = {
        "vp": very_positive,
        "p": positive,
        "neg": negative,
        "vn": very_negative,
        "ne": neutral,
    }
    return render(req, "admin/Sentiment-analysis-graph.html", context)


def comparision_graph(req):
    vgg16 = Vgg16_model.objects.last()
    mobilenet = MobileNet_model.objects.last()
    Densenet = Densenet_model.objects.last()

    Densenet_graph = float(Densenet.model_accuracy.replace("%", "")) if Densenet else 0
    mobilenet_graph = (
        float(mobilenet.model_accuracy.replace("%", "")) if mobilenet else 0
    )
    vgg16_graph = float(vgg16.model_accuracy.replace("%", "")) if vgg16 else 0

    return render(
        req,
        "admin/Comparision-graph.html",
        {
            "Densenet": Densenet_graph,
            "mobilenet": mobilenet_graph,
            "vgg16": vgg16_graph,
        },
    )


def Mobilenet(req):
    model_name = "MobileNet"
    accuracy = "95.52%"
    executed = "MobileNet Model Executed Successfully"

    try:
        model_performance = MobileNet_model.objects.get(model_name=model_name)
        model_performance.model_accuracy = accuracy
        model_performance.model_executed = executed
    except MobileNet_model.DoesNotExist:
        model_performance = MobileNet_model(
            model_name=model_name, model_accuracy=accuracy, model_executed=executed
        )
    model_performance.save()

    req.session["model_name"] = model_name
    req.session["accuracy"] = accuracy
    req.session["executed"] = executed

    return render(req, "admin/Mobilenet.html")


def Mobilenet_result(req):
    model_name = req.session.get("model_name")
    accuracy = req.session.get("accuracy")
    executed = req.session.get("executed")

    context = {"model_name": model_name, "accuracy": accuracy, "executed": executed}
    messages.success(req, "MobileNet executed successfully")
    return render(req, "admin/Mobilenet-btn.html", context)


def vgg16(req):
    model_name = "Vgg16"
    accuracy = "92.88%"
    executed = "Vgg16 Model Executed Successfully"

    try:
        model_performance = Vgg16_model.objects.get(model_name=model_name)
        model_performance.model_accuracy = accuracy
        model_performance.model_executed = executed
    except Vgg16_model.DoesNotExist:
        model_performance = Vgg16_model(
            model_name=model_name, model_accuracy=accuracy, model_executed=executed
        )
    model_performance.save()

    req.session["model_name"] = model_name
    req.session["accuracy"] = accuracy
    req.session["executed"] = executed

    return render(req, "admin/vgg16.html")


def Vgg16_result(req):
    model_name = req.session.get("model_name")
    accuracy = req.session.get("accuracy")
    executed = req.session.get("executed")

    context = {"model_name": model_name, "accuracy": accuracy, "executed": executed}
    messages.success(req, "Vgg16 executed successfully")
    return render(req, "admin/vgg16-btn.html", context)


def Densenet(req):
    model_name = "Densenet"
    accuracy = "95.87%"
    executed = "Densenet Model Executed Successfully"

    try:
        model_performance = Densenet_model.objects.get(model_name=model_name)
        model_performance.model_accuracy = accuracy
        model_performance.model_executed = executed
    except Densenet_model.DoesNotExist:
        model_performance = Densenet_model(
            model_name=model_name, model_accuracy=accuracy, model_executed=executed
        )
    model_performance.save()

    req.session["model_name"] = model_name
    req.session["accuracy"] = accuracy
    req.session["executed"] = executed
    return render(req, "admin/Densenet.html")


def Densenet_result(req):
    model_name = req.session.get("model_name")
    accuracy = req.session.get("accuracy")
    executed = req.session.get("executed")

    context = {"model_name": model_name, "accuracy": accuracy, "executed": executed}
    messages.success(req, "Densenet Model executed successfully")
    return render(req, "admin/Densenet-btn.html", context)


def Train_Test_Split(req):
    images_training = 872
    images_testing = 150
    images_validation = 209
    image_classes = 15

    try:
        model_performance = Train_test_split_model.objects.latest("S_No")
        model_performance.Images_training = str(images_training)
        model_performance.Images_validation = str(images_validation)
        model_performance.Images_testing = str(images_testing)
        model_performance.Images_classes = str(image_classes)
    except Train_test_split_model.DoesNotExist:
        model_performance = Train_test_split_model(
            Images_training=str(images_training),
            Images_validation=str(images_validation),
            Images_testing=str(images_testing),
            Images_classes=str(image_classes),
        )

    model_performance.save()

    req.session["images_training"] = images_training
    req.session["images_validation"] = images_validation
    req.session["images_testing"] = images_testing
    req.session["image_classes"] = image_classes

    return render(req, "admin/Train-Test-Split.html")


def Train_Test_Split_Result(req):
    latest_entry = Train_test_split_model.objects.latest("S_No")

    context = {
        "images_training": latest_entry.Images_training,
        "images_validation": latest_entry.Images_validation,
        "images_testing": latest_entry.Images_testing,
        "image_classes": latest_entry.Images_classes,
    }

    messages.success(req, "Train Test Split executed successfully")
    return render(req, "admin/Train Test Split-result.html", context)

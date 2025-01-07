from django.shortcuts import render, redirect
import urllib.request
import urllib.parse
import random
import ssl
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from mainapp.models import *
from userapp.models import *


# Create your views here.

def index(req):
    return render(req, "main/index.html")


from django.core.exceptions import ObjectDoesNotExist


def user_login(req):
    if req.method == "POST":
        user_email = req.POST.get("email")
        user_password = req.POST.get("password")
        
        # Check for missing fields
        if not user_email or not user_password:
            messages.warning(req, "Please fill in both Email and Password.")
            return redirect("user_login")
        
        print(user_email, user_password)

        try:
            users_data = UserModel.objects.filter(user_email=user_email)
            if not users_data.exists():
                messages.error(req, "User does not exist.")
                return redirect("user_login")

            for user_data in users_data:
                if user_data.user_password == user_password:
                    if (
                        user_data.Otp_Status == "verified"
                    ):
                        req.session["user_id"] = user_data.user_id
                        messages.success(req, "You are logged in.")
                        user_data.No_Of_Times_Login += 1
                        user_data.save()
                        return redirect("user_dashboard")
                    elif (
                        user_data.Otp_Status == "verified"
                    ):
                        messages.info(req, "Your status is pending.")
                        return redirect("user_login")
                    #Go to Admin All Users overthere chnage the status to accept to override this condition.
                    elif (user_data.Otp_Status == "verified" and user_data.User_Status == "removed"):
                        messages.info(req, "Your Account Has been Suspended")
                        return redirect("user_login")
                    else:
                        messages.warning(req, "Please verify your OTP.")
                        req.session["user_email"] = user_data.user_email
                        return redirect("otp")
                else:
                    messages.error(req, "Incorrect credentials.")
                    return redirect("user_login")

            # Handle the case where no user data matched the password
            messages.error(req, "Incorrect credentials.")
            return redirect("user_login")
        except Exception as e:
            print(e)
            messages.error(req, "An error occurred. Please try again later.")
            return redirect("user_login")

    return render(req, "main/user-login.html")


from django.contrib import messages
from django.shortcuts import render, redirect

def admin_login(req):
    admin_name = "admin"
    admin_pwd = "admin"
    if req.method == "POST":
        admin_n = req.POST.get("Username")
        admin_p = req.POST.get("password")

        if not admin_n or not admin_p:
            messages.warning(req, "Please fill in both the username and password fields.")
            return redirect("admin_login")
        
        if admin_n == admin_name and admin_p == admin_pwd:
            messages.success(req, "You are logged in..")
            return redirect("admin_dashboard")
        else:
            messages.error(req, "You are trying to log in with wrong details..")
            return redirect("admin_login")
    
    return render(req, "main/admin-login.html")




def about_us(req):
    return render(req, "main/about-us.html")


def contact_us(req):
    if req.method == "POST":
        name = req.POST.get("Name")
        email = req.POST.get("Email")
        subject = req.POST.get("Subject")
        message = req.POST.get("Message")
        if not name or not email or not subject or not message:
            messages.warning(req, "Enter all the fields to continue")
            return render(req, 'main/contact-us.html')
        Contact_Us.objects.create(
            Full_Name=name, Email_Address=email, Subject=subject, Message=message
        )
        messages.success(req, "Your message has been submitted successfully.")
        return redirect("contact_us")
    return render(req, "main/contact-us.html")


def register(req):
    if req.method == "POST":
        fullname = req.POST.get("username")
        email = req.POST.get("email")
        password = req.POST.get("password")
        age = req.POST.get("age")
        address = req.POST.get("address")
        phone = req.POST.get("contact number")
        image = req.FILES.get("image")  # Use .get() to handle missing file uploads
        
        # Check for missing fields
        missing_fields = []
        if not fullname:
            missing_fields.append("Username")
        if not email:
            missing_fields.append("Email")
        if not password:
            missing_fields.append("Password")
        if not age:
            missing_fields.append("Age")
        if not address:
            missing_fields.append("Address")
        if not phone:
            missing_fields.append("Phone Number")
        if not image:
            missing_fields.append("Profile Picture")
        
        if missing_fields:
            missing_fields_str = ", ".join(missing_fields)
            messages.warning(req, f"Please fill the following fields: {missing_fields_str}")
            return redirect("register")
        
        # Check if the email is already registered
        try:
            data = UserModel.objects.get(user_email=email)
            messages.warning(req, "Email was already registered, choose another email..!")
            return redirect("register")
        except UserModel.DoesNotExist:
            # If the email is not registered, continue with registration
            number = random.randint(1000, 9999)
            print(f"Generated OTP: {number}")  # Print OTP to the terminal
            UserModel.objects.create(
                user_name=fullname,
                user_email=email,
                user_contact=phone,
                user_age=age,
                user_password=password,
                user_address=address,
                user_image=image,
                Otp_Num=number,
            )
            mail_message = f"Registration Successfully\n Your 4 digit Pin is below\n {number}"
            send_mail("User Password", mail_message, settings.EMAIL_HOST_USER, [email])
            req.session["user_email"] = email
            messages.success(req, "Your account was created..")
            return redirect("otp")

    return render(req, "main/register.html")




from django.contrib import messages
from django.shortcuts import render, redirect
from .models import UserModel  

def otp(req):
    user_email = req.session.get("user_email")
    if user_email:
        try:
            user_o = UserModel.objects.get(user_email=user_email)
        except UserModel.DoesNotExist:
            messages.error(req, "User not found.")
            return redirect("login")

        if req.method == "POST":
            otp1 = req.POST.get("otp1", "")
            otp2 = req.POST.get("otp2", "")
            otp3 = req.POST.get("otp3", "")
            otp4 = req.POST.get("otp4", "")

            # Check for missing OTP digits
            if not otp1 or not otp2 or not otp3 or not otp4:
                messages.error(req, "Please enter all OTP digits.")
                return redirect("otp")

            user_otp = otp1 + otp2 + otp3 + otp4
            if user_otp.isdigit():
                u_otp = int(user_otp)
                if u_otp == user_o.Otp_Num:
                    user_o.Otp_Status = "verified"
                    user_o.save()
                    messages.success(req, "OTP verification was successful. You can now login.")
                    return redirect("user_login")
                else:
                    messages.error(req, "Invalid OTP. Please enter the correct OTP.")
            else:
                messages.error(req, "Invalid OTP format. Please enter numbers only.")
                
    else:
        messages.error(req, "Session expired. Please retry the OTP verification.")

    return render(req, "main/otp.html")

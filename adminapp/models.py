from django.db import models


# Create your models here.
class All_users_model(models.Model):
    User_id = models.AutoField(primary_key=True)
    user_Profile = models.FileField(upload_to="images/")
    User_Email = models.EmailField(max_length=50)
    User_Status = models.CharField(max_length=10)

    class Meta:
        db_table = "all_users"


class Densenet_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10)
    model_name = models.CharField(max_length=10)
    model_executed = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "Densenet_model"


class MobileNet_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10)
    model_name = models.CharField(max_length=10)
    model_executed = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "MobileNet_model"


class Vgg16_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    model_accuracy = models.CharField(max_length=10)
    model_name = models.CharField(max_length=10)
    model_executed = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "Vgg16_model"


class Train_test_split_model(models.Model):
    S_No = models.AutoField(primary_key=True)
    Images_training = models.CharField(max_length=10, null=True)
    Images_validation = models.CharField(max_length=10, null=True)
    Images_testing = models.CharField(max_length=10, null=True)
    Images_classes = models.CharField(max_length=10, null=True)

    class Meta:
        db_table = "Traintestsplit"

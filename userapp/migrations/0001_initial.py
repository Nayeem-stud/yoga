# Generated by Django 5.1 on 2024-08-08 20:09

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("mainapp", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Feedback",
            fields=[
                ("Feed_id", models.AutoField(primary_key=True, serialize=False)),
                ("Rating", models.CharField(max_length=100, null=True)),
                ("Review", models.CharField(max_length=225, null=True)),
                ("Sentiment", models.CharField(max_length=100, null=True)),
                ("datetime", models.DateTimeField(auto_now=True)),
                (
                    "Reviewer",
                    models.ForeignKey(
                        null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to="mainapp.usermodel",
                    ),
                ),
            ],
            options={
                "db_table": "feedback_details",
            },
        ),
    ]

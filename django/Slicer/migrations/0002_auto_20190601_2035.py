# Generated by Django 2.2 on 2019-06-01 20:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Slicer', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='seriesinfo',
            name='slicesCnt',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]

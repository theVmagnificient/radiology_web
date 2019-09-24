from django.db import models
import hashlib, os
from datetime import datetime

def HashPassword(passwd):
    passwd += '&^@#&(*~!+)^'
    return hashlib.sha256(passwd.encode()).hexdigest()

class News(models.Model):
    title = models.CharField(max_length=128)
    short_description = models.CharField(max_length=128)
    description = models.CharField(max_length=128)
    time = models.DateTimeField(default=datetime.now, blank=True)
    image_src = models.CharField(max_length=256)

    def __str__(self):
        return self.title

class User(models.Model):
    name = models.CharField(max_length=32)
    surname = models.CharField(max_length=32)
    login = models.CharField(max_length=32)
    mail = models.EmailField(max_length=64)
    passwd = models.CharField(max_length=256)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        ExtendedUser.objects.create(userID=self.id, ava="")

    def __str__(self):
        return "{} {} ({})".format(self.surname, self.name, self.login)

def getAvaName(instance, filename):
    ext = filename.split('.')[-1]
    filename = "{}_{}".format(instance.userID, filename)
    print(os.path.join(instance.standartAvaDir, filename))
    return os.path.join(instance.standartAvaDir, filename)

class ExtendedUser(models.Model):
    userID = models.IntegerField()
    ava = models.ImageField(upload_to=getAvaName)
    standartAvaDir = 'static/avatars/'

    def __str__(self):
        return "Extended User ({})".format(self.userID)

class Chat(models.Model):
    sender = models.IntegerField()
    to = models.IntegerField()
    date = models.DateTimeField(default=datetime.now)
    message = models.CharField(max_length=256)

class Feedback(models.Model):
    user_id = models.IntegerField()
    title = models.CharField(max_length=32)
    text = models.CharField(max_length=1024)
    time = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.title

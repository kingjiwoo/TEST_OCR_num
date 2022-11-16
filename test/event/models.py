from django.db import models

# Create your models here.
class Post(models.Model):
  id = models.AutoField(primary_key= True)
  title = models.CharField(max_length=10)
  image = models.ImageField(upload_to = "images/", null=True, blank=True)

  def __str__(self):
    return str(self.title)

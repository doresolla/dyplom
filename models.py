# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Notes(models.Model):
    note_id = models.AutoField(primary_key=True)
    user = models.ForeignKey('Users', models.DO_NOTHING)
    note_name = models.CharField(max_length=256)
    url = models.CharField(max_length=256)
    file_link = models.CharField(max_length=256)
    created_at = models.DateTimeField(blank=True, null=True)
    def __str__(self):
        return self.note_name
    class Meta:
        managed = False
        db_table = 'notes'


class Users(models.Model):
    user_id = models.AutoField(db_column='user_ID', primary_key=True)  # Field name made lowercase.
    email = models.CharField(max_length=256)
    first_name = models.CharField(max_length=256)
    last_name = models.CharField(max_length=256)
    phone_number = models.CharField(max_length=256)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    def __str__(self):
        return "%s %s" % (self.last_name, self.first_name)
    class Meta:
        managed = False
        db_table = 'users'

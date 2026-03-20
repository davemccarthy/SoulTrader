from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0041_discovery_health"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="description",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="profile",
            name="enabled",
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name="profile",
            name="name",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
    ]

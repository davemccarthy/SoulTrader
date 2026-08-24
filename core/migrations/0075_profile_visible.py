from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0074_discovery_meta"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="visible",
            field=models.BooleanField(default=True),
        ),
    ]

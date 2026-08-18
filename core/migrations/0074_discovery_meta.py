from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0073_profile_sentiment_disabled"),
    ]

    operations = [
        migrations.AddField(
            model_name="discovery",
            name="meta",
            field=models.JSONField(blank=True, default=dict, null=True),
        ),
    ]

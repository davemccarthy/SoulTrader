# Generated manually for AUTO → DISABLED sentiment rename.

from django.db import migrations, models


def forwards_auto_to_disabled(apps, schema_editor):
    Profile = apps.get_model("core", "Profile")
    Profile.objects.filter(sentiment="AUTO").update(sentiment="DISABLED")


def backwards_disabled_to_auto(apps, schema_editor):
    Profile = apps.get_model("core", "Profile")
    Profile.objects.filter(sentiment="DISABLED").update(sentiment="AUTO")


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0072_holding_tranches"),
    ]

    operations = [
        migrations.RunPython(forwards_auto_to_disabled, backwards_disabled_to_auto),
        migrations.AlterField(
            model_name="profile",
            name="sentiment",
            field=models.CharField(
                choices=[
                    ("STRONG_BULL", "STRONG_BULL"),
                    ("BULL", "BULL"),
                    ("STAG", "STAG"),
                    ("BEAR", "BEAR"),
                    ("STRONG_BEAR", "STRONG_BEAR"),
                    ("DISABLED", "DISABLED"),
                ],
                default="DISABLED",
                max_length=16,
            ),
        ),
    ]

# Ensure Noise advisor row exists (register() also creates on module import).

from django.db import migrations


def add_noise_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Noise",
        defaults={
            "name": "Noise",
            "enabled": True,
            "description": "v2 entry: pullback + rel weakness vs QQQ/SPY; no stop-loss.",
        },
    )


def remove_noise_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Noise").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0065_profile_spread_nano"),
    ]

    operations = [
        migrations.RunPython(add_noise_advisor, remove_noise_advisor),
    ]

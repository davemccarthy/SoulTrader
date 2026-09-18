# Ensure Yoyo advisor row exists (register() also creates on module import).

from django.db import migrations


def add_yoyo_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Yoyo",
        defaults={
            "name": "Yoyo",
            "enabled": True,
            "description": (
                "Volatile pack yo-yo: per-name mid-range revive "
                "(price <= mid * (1-X), X from lookback span clipped 4–10%). "
                "48h discovery cooldown; rediscover OK while holding. "
                "PEAKED (15,4) + gated 4% rebuy. Default universe AI-8."
            ),
        },
    )


def remove_yoyo_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Yoyo").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0077_advisor_midway"),
    ]

    operations = [
        migrations.RunPython(add_yoyo_advisor, remove_yoyo_advisor),
    ]

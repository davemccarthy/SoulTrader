# Ensure Midway advisor row exists (register() also creates on module import).

from django.db import migrations


def add_midway_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Midway",
        defaults={
            "name": "Midway",
            "enabled": True,
            "description": (
                "Opportunity book: SO+β quality names soft for market/sentiment "
                "(not company triggers). Regime soft-rank after open+45m; "
                "PEAKED min exit +4% + gated rebuy."
            ),
        },
    )


def remove_midway_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Midway").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0076_advisor_rocket"),
    ]

    operations = [
        migrations.RunPython(add_midway_advisor, remove_midway_advisor),
    ]

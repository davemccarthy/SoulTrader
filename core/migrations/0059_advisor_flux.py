# Ensure Flux advisor row exists (register() also creates on module import).

from django.db import migrations


def add_flux_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Flux",
        defaults={
            "name": "Flux",
            "enabled": True,
            "description": "Averaging-down on a fixed watchlist (WIP).",
        },
    )


def remove_flux_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Flux").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0058_migrate_target_percentage_to_target_price"),
    ]

    operations = [
        migrations.RunPython(add_flux_advisor, remove_flux_advisor),
    ]

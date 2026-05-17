# Re-apply Flux advisor metadata (0059 used get_or_create if an old Flux row existed).

from django.db import migrations


def sync_flux_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Flux",
        defaults={
            "name": "Flux",
            "enabled": True,
            "description": "Averaging-down on a fixed watchlist (WIP).",
        },
    )


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0059_advisor_flux"),
    ]

    operations = [
        migrations.RunPython(sync_flux_advisor, noop_reverse),
    ]

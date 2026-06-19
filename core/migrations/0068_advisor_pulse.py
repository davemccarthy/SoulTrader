# Ensure Pulse advisor row exists (register() also creates on module import).

from django.db import migrations


def add_pulse_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Pulse",
        defaults={
            "name": "Pulse",
            "enabled": True,
            "description": "Daily attention universe: high volume + stable intraday range.",
        },
    )


def remove_pulse_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Pulse").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0067_assessment_so_snapshot"),
    ]

    operations = [
        migrations.RunPython(add_pulse_advisor, remove_pulse_advisor),
    ]

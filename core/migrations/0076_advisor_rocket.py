# Ensure Rocket advisor row exists (register() also creates on module import).

from django.db import migrations


def add_rocket_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Rocket",
        defaults={
            "name": "Rocket",
            "enabled": False,
            "description": (
                "Opening-high tape watcher (CS/ADR gaps ~7.5–15%). "
                "Stage A writes Pending watches; LLM/discover come later."
            ),
        },
    )


def remove_rocket_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Rocket").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0075_profile_visible"),
    ]

    operations = [
        migrations.RunPython(add_rocket_advisor, remove_rocket_advisor),
    ]

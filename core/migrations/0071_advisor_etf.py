# Ensure ETF advisor row exists (register() also creates on module import).

from django.db import migrations


def add_etf_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Etf",
        defaults={
            "name": "ETF",
            "enabled": False,
            "description": (
                "Discovers stocks newly added to tracked thematic ETF holdings "
                "(holdings diff). Exits via profit-flat and optional rebuy on dips."
            ),
        },
    )


def remove_etf_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Etf").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0070_advisor_oracle"),
    ]

    operations = [
        migrations.RunPython(add_etf_advisor, remove_etf_advisor),
    ]

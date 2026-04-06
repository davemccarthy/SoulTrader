# Data migration: PHARM and Bizfeed register() only runs when modules are imported;
# lazy loading + smartanalyse loop skipped them until rows exist. Ensure rows exist.

from django.db import migrations


def add_pharm_bizfeed_advisors(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    for name, python_class in (
        ("PHARM", "Pharm"),
        ("BIZFEED", "Bizfeed"),
    ):
        Advisor.objects.get_or_create(
            python_class=python_class,
            defaults={"name": name, "enabled": True},
        )


def remove_pharm_bizfeed_advisors(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class__in=("Pharm", "Bizfeed")).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0051_profile_sentiment_alter_profile_risk"),
    ]

    operations = [
        migrations.RunPython(add_pharm_bizfeed_advisors, remove_pharm_bizfeed_advisors),
    ]

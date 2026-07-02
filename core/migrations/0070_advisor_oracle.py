# Ensure Oracle advisor row exists (register() also creates on module import).

from django.db import migrations


def add_oracle_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.update_or_create(
        python_class="Oracle",
        defaults={
            "name": "Oracle",
            "enabled": False,
            "description": "Forward pre-earnings scanner: calendar, build, consensus, discover.",
        },
    )


def remove_oracle_advisor(apps, schema_editor):
    Advisor = apps.get_model("core", "Advisor")
    Advisor.objects.filter(python_class="Oracle").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0069_alter_assessment_opp_price_blend_and_more"),
    ]

    operations = [
        migrations.RunPython(add_oracle_advisor, remove_oracle_advisor),
    ]

# MICRO divisor 10 → 15; add NANO (10) for prior MICRO-sized tranches.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0064_remove_assessment_weight"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="spread",
            field=models.CharField(
                blank=True,
                max_length=10,
                null=True,
                choices=[
                    ("MEGA", "MEGA"),
                    ("LARGE", "LARGE"),
                    ("MEDIUM", "MEDIUM"),
                    ("SMALL", "SMALL"),
                    ("MICRO", "MICRO"),
                    ("NANO", "NANO"),
                ],
            ),
        ),
    ]

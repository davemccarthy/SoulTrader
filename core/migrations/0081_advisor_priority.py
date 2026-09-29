# Advisor.priority: discovery run order. 1 = very high (first), 5 = very low (last).
# Existing rows default to Normal (3).

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0080_sellinstruction_profit_cash"),
    ]

    operations = [
        migrations.AddField(
            model_name="advisor",
            name="priority",
            field=models.PositiveSmallIntegerField(
                choices=[
                    (1, "Very high"),
                    (2, "High"),
                    (3, "Normal"),
                    (4, "Low"),
                    (5, "Very low"),
                ],
                default=3,
                help_text="Discovery run order. Very high runs first, Very low last.",
            ),
        ),
    ]

# PERCENTAGE_DOUBLE: same drop/recovery/headline gates as PERCENTAGE_REBUY,
# but add current share count instead of one fund tranche.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0078_advisor_yoyo"),
    ]

    operations = [
        migrations.AlterField(
            model_name="sellinstruction",
            name="instruction",
            field=models.CharField(
                choices=[
                    ("STOP_PRICE", "Stop Loss (Price)"),
                    ("TARGET_PRICE", "Target Price (Price)"),
                    ("STOP_PERCENTAGE", "Stop Loss (Percentage)"),
                    ("TARGET_PERCENTAGE", "Target Price (Percentage)"),
                    ("TARGET_INTRADAY", "Target Intraday"),
                    ("AFTER_DAYS", "After Days (in profit)"),
                    ("DESCENDING_TREND", "Descending trend"),
                    ("END_WEEK", "End of current week"),
                    ("END_DAY", "End of current day"),
                    ("NOT_TRENDING", "No longer trending (low volume)"),
                    ("TARGET_DIMINISHING", "Target Price (Diminishing)"),
                    ("STOP_AUGMENTING", "Stop Loss (Augmenting)"),
                    ("PERCENTAGE_DIMINISHING", "Target Price (Percentage diminishing)"),
                    ("PERCENTAGE_AUGMENTING", "Stop Loss (Percentage augmenting)"),
                    ("PROFIT_TARGET", "Target Profit (Fixed Dollar Amount"),
                    ("PERCENTAGE_REBUY", "Loss - will gamble a Rebuy"),
                    ("PERCENTAGE_DOUBLE", "Loss - double share count"),
                    ("PROFIT_FLAT", "Price flatlined"),
                    ("PEAKED", "Sell when down X% from peak since purchase"),
                ],
                max_length=25,
            ),
        ),
    ]

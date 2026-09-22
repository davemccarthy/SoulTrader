# PROFIT_CASH: flatten when unrealized P&L >= value1 dollars.
# PROFIT_TARGET (Oscilla spend-ratio clipper) is unchanged.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0079_sellinstruction_percentage_double"),
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
                    ("PROFIT_CASH", "Take profit at dollar P&L"),
                    ("PERCENTAGE_REBUY", "Loss - will gamble a Rebuy"),
                    ("PERCENTAGE_DOUBLE", "Loss - double share count"),
                    ("PROFIT_FLAT", "Price flatlined"),
                    ("PEAKED", "Sell when down X% from peak since purchase"),
                ],
                max_length=25,
            ),
        ),
    ]

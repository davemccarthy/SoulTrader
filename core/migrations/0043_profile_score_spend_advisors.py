import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0042_profile_fund_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="advisors",
            field=django.contrib.postgres.fields.ArrayField(
                base_field=models.CharField(max_length=100),
                blank=True,
                default=list,
                size=None,
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="avg_spend",
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=10, null=True),
        ),
        migrations.AddField(
            model_name="profile",
            name="min_score",
            field=models.DecimalField(decimal_places=1, default=30.0, max_digits=5),
        ),
    ]

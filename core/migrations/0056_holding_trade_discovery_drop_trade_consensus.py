from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0055_advisor_description"),
    ]

    operations = [
        migrations.AddField(
            model_name="holding",
            name="discovery",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.DO_NOTHING,
                to="core.discovery",
            ),
        ),
        migrations.AddField(
            model_name="trade",
            name="discovery",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.DO_NOTHING,
                to="core.discovery",
            ),
        ),
        migrations.RemoveField(
            model_name="trade",
            name="consensus",
        ),
    ]

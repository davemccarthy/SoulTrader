from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0043_profile_score_spend_advisors"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="holding",
            name="consensus",
        ),
        migrations.RemoveField(
            model_name="holding",
            name="volatile",
        ),
    ]


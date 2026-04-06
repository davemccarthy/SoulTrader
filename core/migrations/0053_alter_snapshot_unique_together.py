# Snapshot uniqueness must match smartanalyse: one row per (fund, date), not per (user, date).

from django.db import migrations


def dedupe_snapshot_fund_date(apps, schema_editor):
    Snapshot = apps.get_model("core", "Snapshot")
    # Keep the latest row per (fund_id, date); drop fund=NULL rows' dupes by (user_id, date) if needed.
    seen_fund_date = {}
    for snap in Snapshot.objects.order_by("id"):
        fid = snap.fund_id
        if fid is None:
            # Legacy: collapse duplicates on (user, date) before old unique is dropped
            key = ("u", snap.user_id, snap.date)
        else:
            key = ("f", fid, snap.date)
        if key in seen_fund_date:
            snap.delete()
        else:
            seen_fund_date[key] = snap.id


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0052_advisor_pharm_bizfeed"),
    ]

    operations = [
        migrations.RunPython(dedupe_snapshot_fund_date, migrations.RunPython.noop),
        migrations.AlterUniqueTogether(
            name="snapshot",
            unique_together=(),
        ),
        migrations.AlterUniqueTogether(
            name="snapshot",
            unique_together={("fund", "date")},
        ),
    ]

# Holding.tranches: formal buy count for max-tranche rebuy caps.

from decimal import Decimal, ROUND_HALF_UP

from django.db import migrations, models


# Mirror Profile.SPREAD for data migration (historical apps may lack the constant).
SPREAD = {
    "MEGA": 100,
    "LARGE": 60,
    "MEDIUM": 40,
    "SMALL": 25,
    "MICRO": 15,
    "NANO": 10,
}


def backfill_holding_tranches(apps, schema_editor):
    """Count BUY trades since last SELL; fallback to book / average_spend."""
    Holding = apps.get_model("core", "Holding")
    Trade = apps.get_model("core", "Trade")

    updated = 0
    for holding in Holding.objects.select_related("fund").iterator():
        trade_filter = {"stock_id": holding.stock_id, "action": "BUY"}
        sell_filter = {"stock_id": holding.stock_id, "action": "SELL"}
        if holding.fund_id:
            trade_filter["fund_id"] = holding.fund_id
            sell_filter["fund_id"] = holding.fund_id
        else:
            trade_filter["user_id"] = holding.user_id
            sell_filter["user_id"] = holding.user_id

        last_sell = (
            Trade.objects.filter(**sell_filter).order_by("-created", "-id").first()
        )
        buys = Trade.objects.filter(**trade_filter)
        if last_sell and last_sell.created:
            buys = buys.filter(created__gt=last_sell.created)

        count = buys.count()
        if count <= 0:
            count = _fallback_tranches(holding)
        holding.tranches = count
        holding.save(update_fields=["tranches"])
        updated += 1

    print(f"Backfilled tranches for {updated} holdings")


def _fallback_tranches(holding) -> int:
    """Estimate tranche count from cost basis / current average_spend."""
    fund = getattr(holding, "fund", None)
    shares = int(holding.shares or 0)
    avg = holding.average_price or Decimal("0")
    if shares <= 0 or avg <= 0:
        return 1
    if fund is None or not fund.spread or fund.spread not in SPREAD:
        return 1
    num_stocks = Decimal(SPREAD[fund.spread])
    if num_stocks <= 0:
        return 1
    tranche = (fund.investment or Decimal("0")) / num_stocks
    if tranche <= 0:
        return 1
    book = avg * Decimal(shares)
    estimated = int((book / tranche).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(1, estimated)


def reverse_backfill(apps, schema_editor):
    Holding = apps.get_model("core", "Holding")
    Holding.objects.all().update(tranches=0)


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0071_advisor_etf"),
    ]

    operations = [
        migrations.AddField(
            model_name="holding",
            name="tranches",
            field=models.PositiveIntegerField(
                default=0,
                help_text="Number of buys on this open position (initial=1; each rebuy increments).",
            ),
        ),
        migrations.RunPython(backfill_holding_tranches, reverse_backfill),
    ]

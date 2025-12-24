"""
Backfill Stock Metadata Management Command

Usage:
    python manage.py backfill_stock_metadata
    python manage.py backfill_stock_metadata --limit 100
    python manage.py backfill_stock_metadata --missing-only
"""
from django.core.management.base import BaseCommand
from django.db.models import Q
from core.models import Stock
from decimal import Decimal
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Backfill missing stock metadata (sector, industry, exchange, beta) from yfinance'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit the number of stocks to process',
        )
        parser.add_argument(
            '--missing-only',
            action='store_true',
            help='Only process stocks that are missing at least one metadata field',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update all metadata fields even if they already exist',
        )

    def handle(self, *args, **options):
        limit = options['limit']
        missing_only = options['missing_only']
        force = options['force']

        # Build query
        if missing_only:
            # Find stocks missing at least one metadata field
            stocks = Stock.objects.filter(
                Q(sector='') | Q(sector__isnull=True) |
                Q(industry='') | Q(industry__isnull=True) |
                Q(exchange='') | Q(exchange__isnull=True) |
                Q(beta__isnull=True)
            )
        else:
            stocks = Stock.objects.all()

        if limit:
            stocks = stocks[:limit]

        total = stocks.count()
        self.stdout.write(f'Processing {total} stock(s)...')

        updated = 0
        failed = 0
        skipped = 0

        for stock in stocks:
            try:
                # Fetch full info from yfinance
                ticker = yf.Ticker(stock.symbol)
                info = ticker.info

                updated_fields = []

                # Update exchange
                if force or not stock.exchange:
                    exchange = info.get('fullExchangeName') or info.get('exchange')
                    if exchange:
                        stock.exchange = exchange[:32]
                        updated_fields.append('exchange')

                # Update sector
                if force or not stock.sector:
                    sector = info.get('sector')
                    if sector:
                        stock.sector = sector[:100]
                        updated_fields.append('sector')

                # Update industry
                if force or not stock.industry:
                    industry = info.get('industry')
                    if industry:
                        stock.industry = industry[:200]
                        updated_fields.append('industry')

                # Update beta
                if force or stock.beta is None:
                    beta = info.get('beta')
                    if beta is not None:
                        try:
                            stock.beta = Decimal(str(beta))
                            updated_fields.append('beta')
                        except (ValueError, TypeError):
                            pass

                if updated_fields:
                    stock.save()
                    updated += 1
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✓ {stock.symbol}: Updated {", ".join(updated_fields)}'
                        )
                    )
                else:
                    skipped += 1
                    self.stdout.write(
                        f'  {stock.symbol}: No data available from yfinance'
                    )

            except Exception as e:
                failed += 1
                logger.warning(f"Could not update metadata for {stock.symbol}: {e}")
                self.stdout.write(
                    self.style.ERROR(f'✗ {stock.symbol}: {str(e)}')
                )

        # Summary
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(f'Summary:'))
        self.stdout.write(f'  Total processed: {total}')
        self.stdout.write(f'  Updated: {updated}')
        self.stdout.write(f'  Skipped: {skipped}')
        self.stdout.write(f'  Failed: {failed}')


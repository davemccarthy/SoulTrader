"""
Update Stock Prices for Holdings

Usage:
    python manage.py update_stock_prices
"""
import yfinance as yf
from django.core.management.base import BaseCommand
from decimal import Decimal
import logging

from core.models import Stock, Holding

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Update stock prices for all stocks in holdings'
    
    def handle(self, *args, **options):
        # Get all unique stocks that have holdings
        stocks = Stock.objects.filter(holding__isnull=False).distinct()
        
        total = stocks.count()
        updated = 0
        errors = 0
        
        self.stdout.write(f'Updating prices for {total} stocks...')
        
        for stock in stocks:
            try:
                ticker = yf.Ticker(stock.symbol)
                info = ticker.info
                
                # Try multiple price fields (same as Yahoo advisor)
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                if current_price:
                    stock.price = Decimal(str(current_price))
                    stock.save(update_fields=['price', 'updated'])
                    updated += 1
                    self.stdout.write(
                        self.style.SUCCESS(f'✓ Updated {stock.symbol}: ${current_price:.2f}')
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f'⚠ No price found for {stock.symbol}')
                    )
                    errors += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'✗ Error updating {stock.symbol}: {e}')
                )
                errors += 1
                logger.error(f"Error updating price for {stock.symbol}: {e}")
        
        self.stdout.write('')
        self.stdout.write(
            self.style.SUCCESS(f'Completed: {updated} prices updated, {errors} errors out of {total} stocks')
        )


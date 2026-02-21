"""
Run health check for a stock (algorithm + Gemini). Uses existing advisor.health_check().

Usage:
    python manage.py health_check AAPL
    python manage.py health_check TSLA --force   # delete latest health so a fresh check runs
"""
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core.models import Stock, Advisor, SmartAnalysis, Health
from core.services import advisors as advisor_modules


class Command(BaseCommand):
    help = 'Run health check for a stock (confidence score + Gemini weight); prints result.'

    def add_arguments(self, parser):
        parser.add_argument('symbol', type=str, help='Stock symbol (e.g. AAPL)')
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete latest health for this stock so a fresh check is run (otherwise may return cached if < 7 days).',
        )

    def handle(self, *args, **options):
        symbol = options['symbol'].strip().upper()
        force = options.get('force')

        # Resolve stock
        try:
            stock = Stock.objects.get(symbol=symbol)
        except Stock.DoesNotExist:
            advisor_row = Advisor.objects.filter(enabled=True).first()
            if not advisor_row:
                raise CommandError('No enabled advisors found')
            stock = Stock.create(symbol, advisor_row)
            self.stdout.write(f'Created stock {symbol}')
        stock.refresh()

        # First enabled advisor (health_check is on AdvisorBase; any subclass works)
        advisor_row = Advisor.objects.filter(enabled=True).first()
        if not advisor_row:
            raise CommandError('No enabled advisors found')
        module_name = advisor_row.python_class.lower()
        module = getattr(advisor_modules, module_name)
        PythonClass = getattr(module, advisor_row.python_class)
        advisor = PythonClass(advisor_row)

        # Optional: force a fresh run by removing latest health (command-only; no core change)
        if force:
            latest = Health.objects.filter(stock=stock).order_by('-created').first()
            if latest:
                latest.delete()
                self.stdout.write(self.style.WARNING(f'Deleted latest health for {symbol} to force fresh run'))

        # SA required for Health FK (minimal session for testing)
        sa = SmartAnalysis.objects.create(
            username='health_check_command',
            duration=timedelta(),
        )

        try:
            health = advisor.health_check(stock, sa)
        except Exception as e:
            sa.delete()
            raise CommandError(f'health_check failed: {e}')

        if not health:
            sa.delete()
            raise CommandError('health_check returned None')

        # Report
        meta = health.meta or {}
        self.stdout.write(self.style.SUCCESS(f'\nHealth check: {stock.symbol}'))
        self.stdout.write('-' * 60)
        self.stdout.write(f'  Score (final):     {health.score}')
        self.stdout.write(f'  Confidence:       {meta.get("confidence_score")}')
        self.stdout.write(f'  Gemini weight:    {meta.get("gemini_weight")}')
        self.stdout.write(f'  Gemini rec:       {meta.get("gemini_recommendation")}')
        self.stdout.write(f'  Health score:     {meta.get("health_score")}')
        self.stdout.write(f'  Valuation score:  {meta.get("valuation_score")}')
        self.stdout.write(f'  Piotroski:        {meta.get("piotroski")}')
        self.stdout.write(f'  Altman Z:         {meta.get("altman_z")}')
        if meta.get('gemini_explanation'):
            expl = meta['gemini_explanation']
            self.stdout.write(f'  Gemini explanation: {expl[:200]}...' if len(expl) > 200 else f'  Gemini explanation: {expl}')
        self.stdout.write('')

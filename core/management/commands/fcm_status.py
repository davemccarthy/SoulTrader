"""
Report Firebase / FCM configuration as seen by Django (for remote debugging).

Usage:
    python manage.py fcm_status
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from core.services.push import _ensure_firebase_app


class Command(BaseCommand):
    help = 'Print FCM credential source and whether Firebase Admin initializes (no push sent).'

    def handle(self, *args, **options):
        json_blob = (getattr(settings, 'FCM_SERVICE_ACCOUNT_JSON', None) or '').strip()
        cred_path = (getattr(settings, 'FCM_GOOGLE_APPLICATION_CREDENTIALS', None) or '').strip()

        self.stdout.write('FCM / Firebase Admin (server-side)')
        self.stdout.write('-' * 50)
        try:
            import firebase_admin  # noqa: F401
        except ImportError:
            self.stdout.write(self.style.ERROR('  firebase-admin: NOT installed'))
            self.stdout.write('  Install with: pip install firebase-admin')
            return
        self.stdout.write(self.style.SUCCESS('  firebase-admin: installed'))

        if json_blob:
            self.stdout.write(
                f'  FCM_SERVICE_ACCOUNT_JSON: set ({len(json_blob)} chars, not printed)'
            )
        else:
            self.stdout.write('  FCM_SERVICE_ACCOUNT_JSON: (empty)')

        if cred_path:
            p = Path(cred_path).expanduser()
            exists = p.is_file()
            self.stdout.write(f'  FCM/GOOGLE credential path: {p}')
            self.stdout.write(
                self.style.SUCCESS('  file exists: yes')
                if exists
                else self.style.WARNING('  file exists: NO')
            )
        else:
            self.stdout.write('  FCM_GOOGLE_APPLICATION_CREDENTIALS / GOOGLE_APPLICATION_CREDENTIALS: (empty)')

        ok, err = _ensure_firebase_app()
        if ok:
            self.stdout.write(self.style.SUCCESS('  initialize_app: OK'))
        else:
            self.stdout.write(self.style.ERROR(f'  initialize_app: FAILED — {err}'))

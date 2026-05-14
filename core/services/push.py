"""
Firebase Cloud Messaging — send notifications to users' registered devices.

Requires ``pip install firebase-admin`` and ``FCM_GOOGLE_APPLICATION_CREDENTIALS`` (or
``GOOGLE_APPLICATION_CREDENTIALS``) pointing at a Firebase **service account** JSON file.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User

from core.models import PushDevice

logger = logging.getLogger(__name__)

_app_lock = threading.Lock()

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
except ImportError:  # pragma: no cover - optional dependency
    firebase_admin = None
    credentials = None
    messaging = None

# FCM allows up to 500 tokens per multicast request.
_FCM_MULTICAST_LIMIT = 500

# Bundled in the iOS app; used when ``sound`` is omitted.
_DEFAULT_IOS_ALERT_SOUND = 'soultrader.caf'

# Alert title when omitted or empty (match app display name).
_DEFAULT_PUSH_TITLE = 'SOULTRADER'


@dataclass(frozen=True)
class SendPushResult:
    """Outcome of ``push_user`` for one logical send."""

    user_id: int
    device_count: int
    sent: int
    failed: int
    revoked_tokens: int
    skipped: bool
    skip_reason: str


@dataclass(frozen=True)
class SendPushSuperResult:
    """Outcome of ``push_super`` — one logical send to all active superusers."""

    superuser_count: int
    sent: int
    failed: int
    revoked_tokens: int
    skipped: bool
    skip_reason: str


def _ensure_firebase_app() -> tuple[bool, str]:
    """Return (ok, error_message)."""
    if firebase_admin is None:
        return False, 'firebase-admin is not installed (pip install firebase-admin).'

    if firebase_admin._apps:
        return True, ''

    cred_path = getattr(settings, 'FCM_GOOGLE_APPLICATION_CREDENTIALS', '') or ''
    if not cred_path:
        return False, 'FCM_GOOGLE_APPLICATION_CREDENTIALS (or GOOGLE_APPLICATION_CREDENTIALS) is not set.'

    path = Path(cred_path).expanduser()
    if not path.is_file():
        return False, f'Service account file not found: {path}'

    with _app_lock:
        if firebase_admin._apps:
            return True, ''
        try:
            cred = credentials.Certificate(str(path))
            firebase_admin.initialize_app(cred)
        except Exception as exc:  # pragma: no cover - env-specific
            return False, f'Failed to initialize Firebase app: {exc}'
    return True, ''


def _delete_token_if_unregistered(token: str, exc: BaseException) -> bool:
    """Return True if token was removed from DB (invalid / unregistered)."""
    if messaging is None:
        return False
    if isinstance(exc, messaging.UnregisteredError):
        PushDevice.objects.filter(token=token).delete()
        return True
    if isinstance(exc, messaging.SenderIdMismatchError):
        return False
    code = getattr(exc, 'code', None)
    if code == 'NOT_FOUND' or 'registration-token-not-registered' in str(exc).lower():
        PushDevice.objects.filter(token=token).delete()
        return True
    return False


def push_user(user_id: int, title: str, body: str, *, sound: str | None = None) -> SendPushResult:
    """
    Send a notification to all FCM tokens registered for ``user_id``.

    ``sound`` (optional): iOS bundled ``.caf`` filename, e.g. ``\"good_sell.caf\"``.
    If omitted or empty, ``soultrader.caf`` is used (must be in the app bundle).
    Pass ``sound=\"default\"`` for the system default tri-tone.

    If ``title`` is omitted or empty, ``SOULTRADER`` is used.

    Invalid / unregistered tokens are removed from ``PushDevice``.
    """
    title = (title or '').strip() or _DEFAULT_PUSH_TITLE
    body = (body or '').strip()
    if not body:
        return SendPushResult(
            user_id=user_id,
            device_count=0,
            sent=0,
            failed=0,
            revoked_tokens=0,
            skipped=True,
            skip_reason='body is empty',
        )

    tokens = list(
        PushDevice.objects.filter(user_id=user_id).values_list('token', flat=True)
    )
    n = len(tokens)
    if n == 0:
        return SendPushResult(
            user_id=user_id,
            device_count=0,
            sent=0,
            failed=0,
            revoked_tokens=0,
            skipped=True,
            skip_reason='no registered devices',
        )

    ok, err = _ensure_firebase_app()
    if not ok:
        logger.warning('push_user skipped: %s', err)
        return SendPushResult(
            user_id=user_id,
            device_count=n,
            sent=0,
            failed=0,
            revoked_tokens=0,
            skipped=True,
            skip_reason=err,
        )

    sent = 0
    failed = 0
    revoked = 0

    raw = (sound or '').strip()
    if raw.lower() == 'default':
        sound_name = 'default'
    elif raw:
        sound_name = raw
    else:
        sound_name = _DEFAULT_IOS_ALERT_SOUND

    apns_cfg = None
    if messaging is not None:
        apns_cfg = messaging.APNSConfig(
            payload=messaging.APNSPayload(aps=messaging.Aps(sound=sound_name)),
        )

    for start in range(0, n, _FCM_MULTICAST_LIMIT):
        chunk = tokens[start : start + _FCM_MULTICAST_LIMIT]
        msg = messaging.MulticastMessage(
            notification=messaging.Notification(title=title, body=body),
            tokens=chunk,
            apns=apns_cfg,
        )
        batch = messaging.send_each_for_multicast(msg, dry_run=False)
        for i, resp in enumerate(batch.responses):
            token = chunk[i]
            if resp.success:
                sent += 1
            else:
                failed += 1
                exc = resp.exception
                if exc and _delete_token_if_unregistered(token, exc):
                    revoked += 1
                elif exc:
                    logger.info(
                        'FCM send failed user_id=%s token_suffix=…%s: %s',
                        user_id,
                        token[-8:] if len(token) > 8 else token,
                        exc,
                    )

    return SendPushResult(
        user_id=user_id,
        device_count=n,
        sent=sent,
        failed=failed,
        revoked_tokens=revoked,
        skipped=False,
        skip_reason='',
    )


def push_super(
    content: str,
    *,
    title: str = _DEFAULT_PUSH_TITLE,
    sound: str | None = None,
) -> SendPushSuperResult:
    """
    Send a notification to each **active** Django superuser, using ``push_user``
    per user (users with no registered devices simply yield zero sends).

    ``content`` is the notification body. ``title`` defaults to ``SOULTRADER``.
    """
    body = (content or '').strip()
    if not body:
        return SendPushSuperResult(
            superuser_count=0,
            sent=0,
            failed=0,
            revoked_tokens=0,
            skipped=True,
            skip_reason='body is empty',
        )

    super_ids = list(
        User.objects.filter(is_superuser=True, is_active=True).values_list('pk', flat=True)
    )
    n_supers = len(super_ids)
    if n_supers == 0:
        return SendPushSuperResult(
            superuser_count=0,
            sent=0,
            failed=0,
            revoked_tokens=0,
            skipped=True,
            skip_reason='no active superusers',
        )

    alert_title = (title or '').strip() or _DEFAULT_PUSH_TITLE
    sent = 0
    failed = 0
    revoked = 0

    for uid in super_ids:
        r = push_user(uid, alert_title, body, sound=sound)
        sent += r.sent
        failed += r.failed
        revoked += r.revoked_tokens

    return SendPushSuperResult(
        superuser_count=n_supers,
        sent=sent,
        failed=failed,
        revoked_tokens=revoked,
        skipped=False,
        skip_reason='',
    )

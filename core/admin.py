from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.postgres.forms import SimpleArrayField
from django.utils.html import format_html
from django.db.models import Q
from core.models import Advisor, Profile, Watchlist
from decimal import Decimal
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# Register your models here.

@admin.register(Advisor)
class AdvisorAdmin(admin.ModelAdmin):
    list_display = ('name', 'python_class', 'enabled', 'endpoint', 'key')
    list_filter = ('enabled',)
    search_fields = ('name', 'python_class')
    
    # Make name and python_class read-only
    readonly_fields = ('name', 'python_class')
    
    # Disable add and delete actions
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False


# Step 1: Custom User Creation Form with Risk Level dropdown
class UserCreationFormWithRisk(UserCreationForm):
    """Custom user creation form that includes risk level selection from Profile.RISK"""
    
    # Get risk level choices from Profile.RISK keys (single source of truth)
    RISK_CHOICES = [(key, key.replace('_', ' ').title()) for key in Profile.RISK.keys()]
    
    risk_level = forms.ChoiceField(
        choices=RISK_CHOICES,
        initial='MODERATE',
        help_text='Select risk level for the user profile',
        label='Risk Level',
        required=True
    )


# Step 2: Custom UserAdmin that uses the form and creates Profile with risk level
# First, unregister the default User admin
admin.site.unregister(User)

# Now register our custom UserAdmin
@admin.register(User)
class UserAdminWithRisk(BaseUserAdmin):
    """Custom UserAdmin that allows risk level selection during user creation"""
    
    add_form = UserCreationFormWithRisk
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'password1', 'password2', 'risk_level'),
        }),
    )
    
    def get_form(self, request, obj=None, **kwargs):
        """Use custom form for user creation"""
        defaults = {}
        if obj is None:
            defaults['form'] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)
    
    def save_model(self, request, obj, form, change):
        """Override to create Profile with selected risk level"""
        # Save the user first (this will trigger the signal to create Profile with default 'MODERATE')
        super().save_model(request, obj, form, change)
        
        # If this is a new user (not an update), update Profile with risk level from form
        # The signal will have created it already, so we just update it
        if not change and isinstance(form, UserCreationFormWithRisk):
            risk_level = form.cleaned_data.get('risk_level', 'MODERATE')
            Profile.objects.update_or_create(
                user=obj,
                defaults={
                    'risk': risk_level,
                    'cash': Decimal('100000.00'),
                    'investment': Decimal('100000.00')
                }
            )


class ProfileAdminForm(forms.ModelForm):
    advisors = SimpleArrayField(
        forms.CharField(max_length=100),
        required=False,
        help_text='Comma-separated advisor names'
    )

    class Meta:
        model = Profile
        exclude = ('user',)
        help_texts = {
            'sentiment': (
                'Buy/sell sizing and target behavior. Presets map to multipliers in Profile.SENTIMENT; '
                'AUTO is placeholder until liquidity-based calculation is wired.'
            ),
        }


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    form = ProfileAdminForm
    readonly_fields = ('id', 'created')
    list_display = ('id', 'name', 'enabled', 'sentiment', 'investment', 'cash')
    search_fields = ('name', 'description')
    list_filter = ('enabled', 'sentiment')

    fieldsets = (
        ('Profile', {'fields': ('id', 'name', 'description', 'enabled')}),
        ('Strategy', {'fields': ('risk', 'spread', 'sentiment', 'advisors')}),
        ('Capital', {'fields': ('investment', 'cash')}),
    )

    def save_model(self, request, obj, form, change):
        # Keep user hidden in admin: assign creator user for new records.
        if not obj.user_id:
            obj.user = request.user
        super().save_model(request, obj, form, change)


@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    """Admin interface for Watchlist - shows current price and change %"""
    
    list_display = ('symbol', 'company', 'advisor_name', 'watch_price', 'current_price', 'change_percent_display', 'status', 'created', 'expiration_date')
    list_filter = ('status', 'advisor', 'created')
    search_fields = ('stock__symbol', 'stock__company', 'explanation')
    readonly_fields = ('created', 'watch_price', 'current_price_display', 'change_percent_display', 'expiration_date')
    date_hierarchy = 'created'
    
    fieldsets = (
        ('Stock Information', {
            'fields': ('stock', 'advisor', 'status')
        }),
        ('Price Information', {
            'fields': ('price', 'watch_price', 'current_price_display', 'change_percent_display'),
            'description': 'Watch price is the price when added to watchlist. Current price is fetched from market data.'
        }),
        ('Watch Details', {
            'fields': ('explanation', 'days', 'created', 'expiration_date')
        }),
    )
    
    def symbol(self, obj):
        return obj.stock.symbol
    symbol.short_description = 'Symbol'
    symbol.admin_order_field = 'stock__symbol'
    
    def company(self, obj):
        return obj.stock.company or '—'
    company.short_description = 'Company'
    company.admin_order_field = 'stock__company'
    
    def advisor_name(self, obj):
        return obj.advisor.name if obj.advisor else '—'
    advisor_name.short_description = 'Advisor'
    advisor_name.admin_order_field = 'advisor__name'
    
    def watch_price(self, obj):
        return obj.price or Decimal('0')
    watch_price.short_description = 'Watch Price'
    
    def current_price_display(self, obj):
        """Display current price (fetched from yfinance)"""
        current_price = self._get_current_price(obj)
        if current_price:
            return f"${current_price:.2f}"
        return "—"
    current_price_display.short_description = 'Current Price'
    
    def current_price(self, obj):
        """Current price for list display (used for sorting/filtering)"""
        price = self._get_current_price(obj)
        return price or Decimal('0')
    current_price.short_description = 'Current Price'
    current_price.admin_order_field = 'stock__price'
    
    def change_percent_display(self, obj):
        """Display change % with color coding"""
        change_pct = self._calculate_change_percent(obj)
        if change_pct is None:
            return "—"
        
        color = 'green' if change_pct > 0 else 'red' if change_pct < 0 else 'gray'
        sign = '+' if change_pct > 0 else ''
        # Format the percentage as string first, then pass to format_html
        pct_str = f"{sign}{change_pct:.2f}%"
        return format_html(
            '<span style="color: {};">{}</span>',
            color,
            pct_str
        )
    change_percent_display.short_description = 'Change %'
    
    def expiration_date(self, obj):
        """Calculate and display expiration date"""
        from datetime import timedelta
        
        # Calculate: created + days (works with both timezone-aware and naive datetimes)
        expiration = obj.created + timedelta(days=obj.days)
        return expiration
    expiration_date.short_description = 'Expires'
    expiration_date.admin_order_field = 'created'
    
    def _get_current_price(self, obj):
        """Get current price for a watchlist entry (with caching)"""
        # Try to use cached price from stock object
        if obj.stock.price:
            return Decimal(str(obj.stock.price))
        
        # Fallback: fetch from yfinance
        try:
            ticker = yf.Ticker(obj.stock.symbol)
            info = ticker.fast_info
            price = info.get('lastPrice') or info.get('regularMarketPrice')
            if price:
                price_decimal = Decimal(str(price))
                # Update stock cache for future use
                obj.stock.price = price_decimal
                obj.stock.save(update_fields=['price'])
                return price_decimal
        except Exception as e:
            logger.debug(f"Could not fetch price for {obj.stock.symbol}: {e}")
        
        return None
    
    def _calculate_change_percent(self, obj):
        """Calculate change percentage from watch price to current price"""
        watch_price = obj.price
        if not watch_price or watch_price <= 0:
            return None
        
        current_price = self._get_current_price(obj)
        if not current_price:
            return None
        
        change_pct = float((current_price / watch_price) * 100 - 100)
        return change_pct
    
    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        qs = super().get_queryset(request)
        return qs.select_related('stock', 'advisor', 'stock__advisor')
    
    # Batch fetch prices on list view for better performance
    def changelist_view(self, request, extra_context=None):
        """Override to batch fetch prices when viewing list"""
        response = super().changelist_view(request, extra_context)
        
        # If we have a queryset in the response, batch fetch prices
        if hasattr(response, 'context_data') and 'cl' in response.context_data:
            queryset = response.context_data['cl'].queryset
            
            # Get unique symbols
            symbols = list(set(entry.stock.symbol for entry in queryset if entry.stock))
            
            if symbols:
                try:
                    # Batch fetch prices
                    tickers = yf.Tickers(' '.join(symbols))
                    for entry in queryset:
                        if entry.stock:
                            try:
                                ticker = tickers.tickers[entry.stock.symbol]
                                info = ticker.fast_info
                                price = info.get('lastPrice') or info.get('regularMarketPrice')
                                if price:
                                    entry.stock.price = Decimal(str(price))
                                    entry.stock.save(update_fields=['price'])
                            except Exception as e:
                                logger.debug(f"Could not fetch price for {entry.stock.symbol}: {e}")
                    # Refresh queryset to get updated prices
                    response.context_data['cl'].queryset = queryset
                except Exception as e:
                    logger.warning(f"Could not batch fetch prices: {e}")
        
        return response

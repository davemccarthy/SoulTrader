from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django import forms
from core.models import Advisor, Profile
from decimal import Decimal

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

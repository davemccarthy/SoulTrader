from django.contrib import admin
from core.models import Advisor

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

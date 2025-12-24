# Generated manually for sell instruction refactoring

from django.db import migrations, models

def migrate_old_instructions(apps, schema_editor):
    """
    Migrate existing STOP_LOSS to STOP_PERCENTAGE.
    Since the old code already calculated dollar amounts (stock.price * multiplier),
    and the new code also calculates at source, the stored values are already 
    dollar amounts - we just need to rename the instruction type.
    
    Note: TARGET_PRICE remains as-is since it can be either absolute price or
    calculated percentage. Existing records with TARGET_PRICE are already dollar amounts
    from the old calculation, but new ones can be absolute prices.
    """
    SellInstruction = apps.get_model('core', 'SellInstruction')
    
    # Count before migration
    stop_loss_count = SellInstruction.objects.filter(instruction='STOP_LOSS').count()
    
    # Migrate STOP_LOSS -> STOP_PERCENTAGE
    # Old STOP_LOSS records were calculated as: stock.price * multiplier
    # New STOP_PERCENTAGE also calculates at source as: stock.price * multiplier
    # So the stored values are compatible - just rename the type
    SellInstruction.objects.filter(instruction='STOP_LOSS').update(instruction='STOP_PERCENTAGE')
    
    print(f"Migrated {stop_loss_count} STOP_LOSS records -> STOP_PERCENTAGE")

def reverse_migration(apps, schema_editor):
    """Reverse migration - convert back to old type"""
    SellInstruction = apps.get_model('core', 'SellInstruction')
    SellInstruction.objects.filter(instruction='STOP_PERCENTAGE').update(instruction='STOP_LOSS')

class Migration(migrations.Migration):

    dependencies = [
        ('core', '0016_remove_stock_trend'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sellinstruction',
            name='instruction',
            field=models.CharField(
                max_length=20,
                choices=[
                    ('STOP_PRICE', 'Stop Loss (Price)'),
                    ('TARGET_PRICE', 'Target Price (Price)'),
                    ('STOP_PERCENTAGE', 'Stop Loss (Percentage)'),
                    ('TARGET_PERCENTAGE', 'Target Price (Percentage)'),
                    ('CS_FLOOR', 'CS Floor'),
                    ('AFTER_DAYS', 'After Days'),
                    ('DESCENDING_TREND', 'Descending trend'),
                    ('END_WEEK', 'End of current week'),
                    ('END_DAY', 'End of current day'),
                ],
            ),
        ),
        migrations.RunPython(migrate_old_instructions, reverse_migration),
    ]






















































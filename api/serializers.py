from rest_framework import serializers
from django.contrib.auth.models import User
from core.models import PushDevice, Stock, Holding, Trade, Profile


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']


class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['symbol', 'company', 'industry', 'sector', 'exchange', 'price', 'updated']


class HoldingSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)

    class Meta:
        model = Holding
        fields = ['id', 'stock_id', 'stock', 'shares', 'average_price']


class TradeSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)

    class Meta:
        model = Trade
        fields = ['id', 'stock', 'action', 'price', 'shares', 'cost', 'explanation', 'sa', 'created']


class ProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Profile
        fields = ['user', 'risk', 'investment', 'cash']


class PushDeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PushDevice
        fields = ['id', 'token', 'platform', 'environment', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class PushDeviceRegisterSerializer(serializers.Serializer):
    token = serializers.CharField(max_length=512)
    platform = serializers.ChoiceField(choices=PushDevice.Platform.choices)
    environment = serializers.CharField(max_length=16, required=False, allow_blank=True, default='')


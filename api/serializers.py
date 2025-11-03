from rest_framework import serializers
from django.contrib.auth.models import User
from core.models import Stock, Holding, Trade, Profile


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']


class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = ['symbol', 'company', 'exchange', 'price', 'image', 'updated']


class HoldingSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)
    
    class Meta:
        model = Holding
        fields = ['id', 'stock', 'shares', 'average_price', 'volatile']


class TradeSerializer(serializers.ModelSerializer):
    stock = StockSerializer(read_only=True)
    
    class Meta:
        model = Trade
        fields = ['id', 'stock', 'action', 'price', 'shares', 'sa']


class ProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Profile
        fields = ['user', 'risk', 'investment', 'cash']


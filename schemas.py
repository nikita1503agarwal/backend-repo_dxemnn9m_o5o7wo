"""
Database Schemas for Bakery Management System

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name by convention in this project.

Collections:
- User: customers and admins with role-based access
- Product: bakery items with inventory counts
- Ingredient: raw inventory items for production with expiry tracking
- Order: customer orders with items and payment info
- Review: customer reviews for purchased products
- Support: customer support/contact messages

Note: Dates are stored as ISO 8601 strings in requests but converted to
Python datetime objects when inserted via database helpers.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime


class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    contact: Optional[str] = Field(None, description="Phone number")
    hashed_password: str = Field(..., description="BCrypt hashed password")
    role: str = Field("customer", description="Role: customer or admin")
    is_active: bool = Field(True, description="Whether user is active")


class Product(BaseModel):
    name: str = Field(..., description="Product name")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Unit price")
    image_url: Optional[str] = Field(None, description="Image URL")
    category: str = Field(..., description="Category e.g., cakes, bread, pastries")
    stock_qty: int = Field(0, ge=0, description="Available product quantity for sale")
    is_active: bool = Field(True, description="Product available for listing")


class Ingredient(BaseModel):
    name: str = Field(..., description="Ingredient name")
    quantity: float = Field(..., ge=0, description="Quantity in stock")
    unit: str = Field(..., description="Measurement unit, e.g., kg, g, l, pcs")
    low_threshold: float = Field(0, ge=0, description="Low stock alert threshold")
    expiry_date: Optional[datetime] = Field(None, description="Expiry date if perishable")


class OrderItem(BaseModel):
    product_id: str = Field(..., description="ID of the product")
    name: str = Field(..., description="Snapshot of product name at purchase time")
    unit_price: float = Field(..., ge=0, description="Snapshot of price at purchase time")
    quantity: int = Field(..., ge=1, description="Quantity ordered")


class PaymentInfo(BaseModel):
    method: str = Field(..., description="payment method: cod, card, wallet, netbanking")
    status: str = Field("pending", description="payment status: pending, paid, failed, refunded")
    transaction_id: Optional[str] = Field(None, description="External gateway transaction reference")


class Order(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    user_id: str = Field(..., description="ID of the customer")
    items: List[OrderItem] = Field(default_factory=list)
    subtotal: float = Field(..., ge=0)
    tax: float = Field(0, ge=0)
    total: float = Field(..., ge=0)
    delivery_mode: str = Field(..., description="delivery or pickup")
    status: str = Field("placed", description="Order status: placed, processing, out_for_delivery, delivered, cancelled")
    payment: PaymentInfo


class Review(BaseModel):
    product_id: str = Field(..., description="ID of the reviewed product")
    user_id: str = Field(..., description="ID of the reviewer")
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    approved: bool = Field(False, description="Admin moderated flag")


class Support(BaseModel):
    user_id: Optional[str] = None
    name: str
    email: EmailStr
    message: str

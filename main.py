import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

from database import db, create_document, get_documents
from schemas import User as UserSchema, Product as ProductSchema, Ingredient as IngredientSchema, Order as OrderSchema, Review as ReviewSchema, Support as SupportSchema

from bson import ObjectId

# ---------- Auth setup ----------
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None


# ---------- FastAPI app ----------
app = FastAPI(title="Bakery Management System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Utility helpers ----------
class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailStr
    contact: Optional[str]
    role: str


def oid(s: str) -> ObjectId:
    try:
        return ObjectId(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")


def doc_to_public_user(doc) -> UserPublic:
    return UserPublic(
        id=str(doc.get("_id")),
        name=doc.get("name"),
        email=doc.get("email"),
        contact=doc.get("contact"),
        role=doc.get("role", "customer"),
    )


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, role=role)
    except JWTError:
        raise credentials_exception

    user = db["user"].find_one({"_id": oid(token_data.user_id)})
    if not user:
        raise credentials_exception
    return user


def require_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


# ---------- Health ----------
@app.get("/")
def read_root():
    return {"message": "Bakery Management System API"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            collections = db.list_collection_names()
            response["collections"] = collections
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# ---------- Auth Endpoints ----------
class RegisterPayload(BaseModel):
    name: str
    email: EmailStr
    contact: Optional[str] = None
    password: str
    role: Optional[str] = "customer"


class LoginPayload(BaseModel):
    email: EmailStr
    password: str


@app.post("/auth/register", response_model=UserPublic)
def register(payload: RegisterPayload):
    if db["user"].find_one({"email": payload.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(payload.password)
    user_doc = UserSchema(
        name=payload.name,
        email=payload.email,
        contact=payload.contact,
        hashed_password=hashed,
        role="admin" if payload.role == "admin" else "customer",
        is_active=True,
    ).model_dump()
    user_id = create_document("user", user_doc)
    doc = db["user"].find_one({"_id": ObjectId(user_id)})
    return doc_to_public_user(doc)


@app.post("/auth/login", response_model=Token)
def login(payload: LoginPayload):
    user = db["user"].find_one({"email": payload.email})
    if not user or not verify_password(payload.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user["_id"]), "role": user.get("role", "customer")})
    return Token(access_token=access_token)


@app.get("/auth/me", response_model=UserPublic)
def me(user=Depends(get_current_user)):
    return doc_to_public_user(user)


# ---------- Product Management ----------
@app.get("/products")
def list_products(category: Optional[str] = None):
    query = {"is_active": True}
    if category:
        query["category"] = category
    items = db["product"].find(query)
    result = []
    for it in items:
        it["id"] = str(it.pop("_id"))
        result.append(it)
    return result


@app.post("/products")
def create_product(product: ProductSchema, admin=Depends(require_admin)):
    pid = create_document("product", product)
    doc = db["product"].find_one({"_id": ObjectId(pid)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.put("/products/{product_id}")
def update_product(product_id: str, product: ProductSchema, admin=Depends(require_admin)):
    data = product.model_dump()
    data["updated_at"] = datetime.now(timezone.utc)
    res = db["product"].update_one({"_id": oid(product_id)}, {"$set": data})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    doc = db["product"].find_one({"_id": oid(product_id)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.delete("/products/{product_id}")
def delete_product(product_id: str, admin=Depends(require_admin)):
    res = db["product"].delete_one({"_id": oid(product_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"deleted": True}


# ---------- Ingredient / Inventory ----------
@app.get("/ingredients")
def list_ingredients(admin=Depends(require_admin)):
    docs = db["ingredient"].find()
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out


@app.post("/ingredients")
def create_ingredient(item: IngredientSchema, admin=Depends(require_admin)):
    iid = create_document("ingredient", item)
    doc = db["ingredient"].find_one({"_id": ObjectId(iid)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.put("/ingredients/{ingredient_id}")
def update_ingredient(ingredient_id: str, item: IngredientSchema, admin=Depends(require_admin)):
    res = db["ingredient"].update_one({"_id": oid(ingredient_id)}, {"$set": item.model_dump()})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    doc = db["ingredient"].find_one({"_id": oid(ingredient_id)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.get("/inventory/low-stock")
def low_stock_alerts(admin=Depends(require_admin)):
    docs = db["ingredient"].find({"$expr": {"$lt": ["$quantity", "$low_threshold"]}})
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out


# ---------- Orders ----------
class CartItem(BaseModel):
    product_id: str
    quantity: int


class CreateOrderPayload(BaseModel):
    items: List[CartItem]
    delivery_mode: str
    payment_method: str = "cod"


@app.post("/orders")
def place_order(payload: CreateOrderPayload, user=Depends(get_current_user)):
    if not payload.items:
        raise HTTPException(status_code=400, detail="Cart is empty")

    order_items = []
    subtotal = 0.0

    for ci in payload.items:
        prod = db["product"].find_one({"_id": oid(ci.product_id)})
        if not prod or not prod.get("is_active", True):
            raise HTTPException(status_code=400, detail="Product unavailable")
        if int(prod.get("stock_qty", 0)) < ci.quantity:
            raise HTTPException(status_code=400, detail=f"Insufficient stock for {prod.get('name')}")
        order_items.append({
            "product_id": ci.product_id,
            "name": prod.get("name"),
            "unit_price": float(prod.get("price", 0)),
            "quantity": ci.quantity,
        })
        subtotal += float(prod.get("price", 0)) * ci.quantity

    tax = round(subtotal * 0.05, 2)
    total = round(subtotal + tax, 2)

    order_id = f"ORD-{int(datetime.now(timezone.utc).timestamp())}"
    order_doc = OrderSchema(
        order_id=order_id,
        user_id=str(user["_id"]),
        items=[item for item in [
            # Validate via Pydantic
        ]],
        subtotal=subtotal,
        tax=tax,
        total=total,
        delivery_mode=payload.delivery_mode,
        status="placed",
        payment={"method": payload.payment_method, "status": "pending", "transaction_id": None},
    )
    # Convert to dict and override items with validated OrderItem dicts
    order_dict = order_doc.model_dump()
    order_dict["items"] = order_items

    # Insert order
    oid_str = create_document("order", order_dict)

    # Decrement stock
    for it in payload.items:
        db["product"].update_one({"_id": oid(it.product_id)}, {"$inc": {"stock_qty": -it.quantity}})

    doc = db["order"].find_one({"_id": ObjectId(oid_str)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.get("/orders/me")
def my_orders(user=Depends(get_current_user)):
    docs = db["order"].find({"user_id": str(user["_id"])})
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out


@app.get("/orders")
def list_orders(admin=Depends(require_admin)):
    docs = db["order"].find().sort("created_at", -1)
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out


class UpdateOrderStatus(BaseModel):
    status: str
    payment_status: Optional[str] = None


@app.put("/orders/{order_id}")
def update_order(order_id: str, payload: UpdateOrderStatus, admin=Depends(require_admin)):
    update = {"status": payload.status}
    if payload.payment_status:
        update["payment.status"] = payload.payment_status
    res = db["order"].update_one({"_id": oid(order_id)}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    doc = db["order"].find_one({"_id": oid(order_id)})
    doc["id"] = str(doc.pop("_id"))
    return doc


# ---------- Reviews ----------
@app.post("/reviews")
def create_review(review: ReviewSchema, user=Depends(get_current_user)):
    if review.user_id != str(user["_id"]):
        raise HTTPException(status_code=403, detail="Cannot post review for another user")
    rid = create_document("review", review)
    doc = db["review"].find_one({"_id": ObjectId(rid)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.get("/reviews/{product_id}")
def list_reviews(product_id: str):
    docs = db["review"].find({"product_id": product_id, "approved": True})
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out


@app.put("/reviews/{review_id}/approve")
def approve_review(review_id: str, admin=Depends(require_admin)):
    res = db["review"].update_one({"_id": oid(review_id)}, {"$set": {"approved": True}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"approved": True}


# ---------- Reports ----------
@app.get("/reports/sales")
def sales_report(admin=Depends(require_admin)):
    total_orders = db["order"].count_documents({})
    total_revenue = 0.0
    for d in db["order"].find({}):
        total_revenue += float(d.get("total", 0))
    top_products = list(db["order"].aggregate([
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.product_id", "qty": {"$sum": "$items.quantity"}}},
        {"$sort": {"qty": -1}},
        {"$limit": 5},
    ]))
    # convert ids
    for t in top_products:
        t["product_id"] = t.pop("_id")
    return {
        "total_orders": total_orders,
        "total_revenue": round(total_revenue, 2),
        "top_products": top_products,
    }


@app.get("/reports/inventory")
def inventory_report(admin=Depends(require_admin)):
    total_products = db["product"].count_documents({})
    low_stock = db["ingredient"].count_documents({"$expr": {"$lt": ["$quantity", "$low_threshold"]}})
    return {"total_products": total_products, "low_stock_ingredients": low_stock}


# ---------- Support & FAQs ----------
@app.post("/support")
def create_support(msg: SupportSchema):
    sid = create_document("support", msg)
    doc = db["support"].find_one({"_id": ObjectId(sid)})
    doc["id"] = str(doc.pop("_id"))
    return doc


@app.get("/support", dependencies=[Depends(require_admin)])
def list_support():
    docs = db["support"].find().sort("created_at", -1)
    out = []
    for d in docs:
        d["id"] = str(d.pop("_id"))
        out.append(d)
    return out

"""测试数据生成脚本

用于生成电商系统的测试数据，包括品牌、分类、商品、用户和订单。
"""

import random
from decimal import Decimal

from sqlalchemy.orm import Session

from .engine import get_db_session
from .models import Brand, MainCategory, SubCategory, Product, User, Order, OrderItem
from .crud import create_order, get_or_create_user


# 测试数据
BRANDS_DATA = [
    {"name": "Apple", "image": "/images/brands/apple.png"},
    {"name": "华为", "image": "/images/brands/huawei.png"},
    {"name": "小米", "image": "/images/brands/xiaomi.png"},
    {"name": "OPPO", "image": "/images/brands/oppo.png"},
    {"name": "vivo", "image": "/images/brands/vivo.png"},
    {"name": "三星", "image": "/images/brands/samsung.png"},
]

CATEGORIES_DATA = {
    "Apple": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机"],
        },
        {
            "main": "电脑",
            "subs": ["笔记本电脑", "平板电脑"],
        },
    ],
    "华为": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机", "折叠屏手机"],
        },
        {
            "main": "电脑",
            "subs": ["笔记本电脑", "平板电脑"],
        },
    ],
    "小米": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机"],
        },
        {
            "main": "家电",
            "subs": ["电视", "空调", "洗衣机"],
        },
    ],
    "OPPO": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机"],
        },
    ],
    "vivo": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机"],
        },
    ],
    "三星": [
        {
            "main": "手机",
            "subs": ["智能手机", "5G手机", "折叠屏手机"],
        },
        {
            "main": "电脑",
            "subs": ["笔记本电脑"],
        },
    ],
}

PRODUCTS_DATA = [
    # Apple
    {"name": "iPhone 15 Pro", "brand": "Apple", "main_cat": "手机", "sub_cat": "智能手机", "model": "A2848", "price": 7999, "stock": 50, "rating": 4.8},
    {"name": "iPhone 15", "brand": "Apple", "main_cat": "手机", "sub_cat": "5G手机", "model": "A2849", "price": 5999, "stock": 80, "rating": 4.7},
    {"name": "MacBook Pro 14", "brand": "Apple", "main_cat": "电脑", "sub_cat": "笔记本电脑", "model": "MK2H3", "price": 14999, "stock": 30, "rating": 4.9},
    {"name": "iPad Pro 12.9", "brand": "Apple", "main_cat": "电脑", "sub_cat": "平板电脑", "model": "MWTX3", "price": 8499, "stock": 40, "rating": 4.8},
    # 华为
    {"name": "华为 Mate 60 Pro", "brand": "华为", "main_cat": "手机", "sub_cat": "5G手机", "model": "ALN-AL00", "price": 6999, "stock": 60, "rating": 4.9},
    {"name": "华为 P60 Pro", "brand": "华为", "main_cat": "手机", "sub_cat": "智能手机", "model": "ELE-AL00", "price": 5988, "stock": 45, "rating": 4.7},
    {"name": "华为 Mate X5", "brand": "华为", "main_cat": "手机", "sub_cat": "折叠屏手机", "model": "ALT-AL00", "price": 12999, "stock": 15, "rating": 4.8},
    {"name": "华为 MateBook X Pro", "brand": "华为", "main_cat": "电脑", "sub_cat": "笔记本电脑", "model": "KPLVW", "price": 8999, "stock": 25, "rating": 4.6},
    # 小米
    {"name": "小米14 Pro", "brand": "小米", "main_cat": "手机", "sub_cat": "5G手机", "model": "2312DRA50C", "price": 4999, "stock": 100, "rating": 4.7},
    {"name": "红米 K70 Pro", "brand": "小米", "main_cat": "手机", "sub_cat": "智能手机", "model": "23113RKC6C", "price": 3299, "stock": 150, "rating": 4.6},
    {"name": "小米电视 75 Pro", "brand": "小米", "main_cat": "家电", "sub_cat": "电视", "model": "L75M7-EA", "price": 4999, "stock": 35, "rating": 4.5},
    {"name": "米家空调 1.5匹", "brand": "小米", "main_cat": "家电", "sub_cat": "空调", "model": "KFR-35GW", "price": 2199, "stock": 80, "rating": 4.4},
    # OPPO
    {"name": "OPPO Find X7 Pro", "brand": "OPPO", "main_cat": "手机", "sub_cat": "5G手机", "model": "PHY110", "price": 5999, "stock": 55, "rating": 4.7},
    {"name": "OPPO Reno12 Pro", "brand": "OPPO", "main_cat": "手机", "sub_cat": "智能手机", "model": "PJW110", "price": 3699, "stock": 90, "rating": 4.5},
    # vivo
    {"name": "vivo X100 Pro", "brand": "vivo", "main_cat": "手机", "sub_cat": "5G手机", "model": "V2330A", "price": 5499, "stock": 70, "rating": 4.7},
    {"name": "vivo S18 Pro", "brand": "vivo", "main_cat": "手机", "sub_cat": "智能手机", "model": "V2336A", "price": 3199, "stock": 120, "rating": 4.6},
    # 三星
    {"name": "三星 Galaxy S24 Ultra", "brand": "三星", "main_cat": "手机", "sub_cat": "5G手机", "model": "SM-S9280", "price": 9699, "stock": 40, "rating": 4.8},
    {"name": "三星 Galaxy Z Fold5", "brand": "三星", "main_cat": "手机", "sub_cat": "折叠屏手机", "model": "SM-F9460", "price": 12999, "stock": 20, "rating": 4.7},
    {"name": "三星 Galaxy Book3 Pro", "brand": "三星", "main_cat": "电脑", "sub_cat": "笔记本电脑", "model": "NP960XFH", "price": 7999, "stock": 30, "rating": 4.5},
]

USERS_DATA = [
    {"phone": "13800138000", "name": "张三"},
    {"phone": "13800138001", "name": "李四"},
    {"phone": "13800138002", "name": "王五"},
]


def seed_brands(db: Session) -> dict:
    """创建品牌数据"""
    brand_map = {}
    for brand_data in BRANDS_DATA:
        # 检查是否已存在
        existing = db.execute(
            select(Brand).where(Brand.name == brand_data["name"])
        ).scalar_one_or_none()
        if existing:
            brand_map[brand_data["name"]] = existing
        else:
            brand = Brand(**brand_data)
            db.add(brand)
            db.flush()
            brand_map[brand["name"]] = brand
    return brand_map


def seed_categories(db: Session, brand_map: dict) -> dict:
    """创建分类数据"""
    from .crud import get_main_category_by_name, get_sub_category_by_name

    category_map = {}  # (brand, main_cat, sub_cat) -> object

    for brand_name, categories in CATEGORIES_DATA.items():
        brand = brand_map[brand_name]
        for cat_data in categories:
            main_cat_name = cat_data["main"]

            # 检查/创建主分类
            main_cat = get_main_category_by_name(db, main_cat_name)
            if not main_cat:
                main_cat = MainCategory(name=main_cat_name, brand_id=brand.id)
                db.add(main_cat)
                db.flush()

            # 创建子分类
            for sub_cat_name in cat_data["subs"]:
                sub_cat = get_sub_category_by_name(db, sub_cat_name)
                if not sub_cat:
                    sub_cat = SubCategory(name=sub_cat_name, main_category_id=main_cat.id)
                    db.add(sub_cat)
                    db.flush()

                category_map[(brand_name, main_cat_name, sub_cat_name)] = {
                    "main": main_cat,
                    "sub": sub_cat,
                }

    return category_map


def seed_products(db: Session, brand_map: dict, category_map: dict) -> list:
    """创建商品数据"""
    from .crud import get_product_by_name

    products = []
    for prod_data in PRODUCTS_DATA:
        # 检查是否已存在
        existing = db.execute(
            select(Product).where(Product.name == prod_data["name"])
        ).scalar_one_or_none()
        if existing:
            products.append(existing)
            continue

        brand = brand_map[prod_data["brand"]]
        cat_key = (prod_data["brand"], prod_data["main_cat"], prod_data["sub_cat"])
        cats = category_map.get(cat_key)

        if not cats:
            print(f"Warning: Category not found for {prod_data['name']}")
            continue

        product = Product(
            name=prod_data["name"],
            model_number=prod_data["model"],
            brand_id=brand.id,
            main_category_id=cats["main"].id,
            sub_category_id=cats["sub"].id,
            price=Decimal(str(prod_data["price"])),
            stock=prod_data["stock"],
            rating=prod_data["rating"],
            review_count=random.randint(100, 5000),
            special=random.choice([True, False]),
        )
        db.add(product)
        db.flush()
        products.append(product)

    return products


def seed_users(db: Session) -> list:
    """创建用户数据"""
    from .crud import get_user_by_phone

    users = []
    for user_data in USERS_DATA:
        user = get_user_by_phone(db, user_data["phone"])
        if not user:
            user = User(**user_data)
            db.add(user)
            db.flush()
        users.append(user)
    return users


def seed_orders(db: Session, users: list, products: list) -> list:
    """创建订单数据"""
    orders = []
    for user in users:
        # 每个用户创建 1-3 个订单
        for _ in range(random.randint(1, 3)):
            # 随机选择 1-3 个商品
            items = []
            for _ in range(random.randint(1, 3)):
                product = random.choice(products)
                items.append({
                    "product_id": product.id,
                    "quantity": random.randint(1, 2),
                })

            try:
                order = create_order(
                    db,
                    user_id=user.id,
                    items=items,
                    notes=random.choice(["", "尽快发货", "周末配送"]),
                )

                # 随机设置订单状态
                order.status = random.choice([
                    "pending", "paid", "shipped", "delivered", "cancelled"
                ])
                db.flush()
                orders.append(order)
            except Exception as e:
                print(f"Warning: Failed to create order: {e}")

    return orders


def seed_all(drop_existing: bool = False) -> None:
    """生成所有测试数据

    Args:
        drop_existing: 是否删除现有数据
    """
    with get_db_session() as db:
        if drop_existing:
            print("Dropping existing data...")
            db.execute("DELETE FROM order_items")
            db.execute("DELETE FROM orders")
            db.execute("DELETE FROM products")
            db.execute("DELETE FROM sub_categories")
            db.execute("DELETE FROM main_categories")
            db.execute("DELETE FROM brands")
            db.execute("DELETE FROM users")
            db.commit()

        print("Seeding brands...")
        brand_map = seed_brands(db)

        print("Seeding categories...")
        category_map = seed_categories(db, brand_map)

        print("Seeding products...")
        products = seed_products(db, brand_map, category_map)

        print("Seeding users...")
        users = seed_users(db)

        print("Seeding orders...")
        orders = seed_orders(db, users, products)

        db.commit()

        print(f"\nSeeding completed!")
        print(f"  - Brands: {len(brand_map)}")
        print(f"  - Products: {len(products)}")
        print(f"  - Users: {len(users)}")
        print(f"  - Orders: {len(orders)}")


if __name__ == "__main__":
    # 导入 select
    from sqlalchemy import select

    seed_all(drop_existing=False)

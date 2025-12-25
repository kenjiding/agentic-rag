"""数据库连接管理

提供 PostgreSQL 连接配置和会话管理。
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

# 数据库 URL 配置
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://root:Chengyi123456&@chengyi-server.chogc8iyy9og.eu-central-1.rds.amazonaws.com:5432/chengyi?sslmode=require"
)

# 创建引擎
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # 连接健康检查
    pool_size=5,
    max_overflow=10,
    echo=False,  # 设置为 True 可查看 SQL 日志
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(create_tables: bool = False) -> None:
    """初始化数据库

    Args:
        create_tables: 是否创建表结构（生产环境谨慎使用）
    """
    if create_tables:
        Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """获取数据库会话

    用于在非 Web 环境下直接获取会话。

    Returns:
        Session: SQLAlchemy 会话对象
    """
    return SessionLocal()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器

    自动处理事务提交/回滚和会话关闭。

    Yields:
        Session: SQLAlchemy 会话对象

    Example:
        with get_db_session() as db:
            products = db.query(Product).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def test_connection() -> bool:
    """测试数据库连接

    Returns:
        bool: 连接成功返回 True
    """
    try:
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

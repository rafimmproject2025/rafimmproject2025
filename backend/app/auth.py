import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
import psycopg
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/candidate_news")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
DEFAULT_ADMIN_USER = os.getenv("DEFAULT_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")

# Use pbkdf2_sha256 to avoid bcrypt backend issues/length limits
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def _get_conn():
    return await psycopg.AsyncConnection.connect(DB_DSN)


def _now():
    return datetime.now(timezone.utc)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify((plain_password or ""), hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return pwd_context.hash((password or ""))


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = _now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def _fetch_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT user_id, username, password_hash, role::text, is_active, display_name, created_at, updated_at, last_login
            FROM app_user
            WHERE LOWER(username) = LOWER(%s)
            """,
            (username,),
        )
        row = await cur.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "role": row[3],
        "is_active": bool(row[4]),
        "display_name": row[5],
        "created_at": row[6],
        "updated_at": row[7],
        "last_login": row[8],
    }


async def _fetch_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT user_id, username, password_hash, role::text, is_active, display_name, created_at, updated_at, last_login
            FROM app_user
            WHERE user_id = %s
            """,
            (user_id,),
        )
        row = await cur.fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "role": row[3],
        "is_active": bool(row[4]),
        "display_name": row[5],
        "created_at": row[6],
        "updated_at": row[7],
        "last_login": row[8],
    }


def _public_user(u: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": u.get("user_id"),
        "username": u.get("username"),
        "role": u.get("role"),
        "is_active": u.get("is_active", True),
        "display_name": u.get("display_name"),
        "created_at": u.get("created_at"),
        "updated_at": u.get("updated_at"),
        "last_login": u.get("last_login"),
    }


async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = await _fetch_user_by_username(username)
    if not user or not user.get("is_active", True):
        return None
    if not verify_password(password, user.get("password_hash") or ""):
        return None
    return user


async def ensure_default_admin():
    if not DEFAULT_ADMIN_USER or not DEFAULT_ADMIN_PASSWORD:
        return
    existing = await _fetch_user_by_username(DEFAULT_ADMIN_USER)
    if existing:
        if existing.get("role") != "admin":
            async with await _get_conn() as con:
                await con.execute(
                    "UPDATE app_user SET role='admin'::app_role, is_active=TRUE WHERE user_id=%s",
                    (existing["user_id"],),
                )
        return
    hashed = get_password_hash(DEFAULT_ADMIN_PASSWORD)
    async with await _get_conn() as con:
        await con.execute(
            """
            INSERT INTO app_user(username, password_hash, role, display_name, is_active, created_at, updated_at)
            VALUES (%s, %s, 'admin', %s, TRUE, %s, %s)
            ON CONFLICT (LOWER(username)) DO NOTHING
            """,
            (DEFAULT_ADMIN_USER, hashed, "Default Admin", _now(), _now()),
        )


async def issue_token_for_user(user: Dict[str, Any]) -> str:
    payload = {"sub": user.get("username"), "role": user.get("role"), "uid": user.get("user_id")}
    return create_access_token(payload)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if SERVICE_TOKEN and token == SERVICE_TOKEN:
        return {"user_id": 0, "username": "service", "role": "admin", "is_active": True}
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    user = await _fetch_user_by_username(username)
    if user is None or not user.get("is_active", True):
        raise credentials_exception
    return user


async def require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if (user.get("role") or "").lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


async def list_users() -> list[Dict[str, Any]]:
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            SELECT user_id, username, role::text, is_active, display_name, created_at, updated_at, last_login
            FROM app_user
            ORDER BY user_id ASC
            """
        )
        rows = await cur.fetchall()
    return [
        {
            "user_id": r[0],
            "username": r[1],
            "role": r[2],
            "is_active": bool(r[3]),
            "display_name": r[4],
            "created_at": r[5],
            "updated_at": r[6],
            "last_login": r[7],
        }
        for r in rows
    ]


async def create_user(
    username: str,
    password: str,
    role: str = "user",
    display_name: Optional[str] = None,
    is_active: bool = True,
) -> Dict[str, Any]:
    role_val = (role or "user").lower()
    if role_val not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Invalid role")
    hashed = get_password_hash(password)
    now_ts = _now()
    async with await _get_conn() as con:
        cur = await con.execute(
            """
            INSERT INTO app_user(username, password_hash, role, display_name, is_active, created_at, updated_at)
            VALUES (%s, %s, %s::app_role, %s, %s, %s, %s)
            ON CONFLICT (LOWER(username)) DO NOTHING
            RETURNING user_id
            """,
            (username, hashed, role_val, display_name, is_active, now_ts, now_ts),
        )
        row = await cur.fetchone()
        if not row:
            existing = await _fetch_user_by_username(username)
            if existing:
                raise HTTPException(status_code=409, detail="User already exists")
            raise HTTPException(status_code=400, detail="Could not create user")
        user_id = row[0]
    created = await _fetch_user_by_id(user_id)
    return _public_user(created)


async def update_user(
    user_id: int,
    *,
    display_name: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    fields = []
    params: list[Any] = []
    if display_name is not None:
        fields.append("display_name = %s")
        params.append(display_name)
    if role is not None:
        role_val = (role or "").lower()
        if role_val not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="Invalid role")
        fields.append("role = %s::app_role")
        params.append(role_val)
    if is_active is not None:
        fields.append("is_active = %s")
        params.append(bool(is_active))
    if password is not None:
        fields.append("password_hash = %s")
        params.append(get_password_hash(password))
    fields.append("updated_at = %s")
    params.append(_now())
    if not fields:
        raise HTTPException(status_code=400, detail="No updates provided")

    params.append(int(user_id))
    async with await _get_conn() as con:
        cur = await con.execute(
            f"UPDATE app_user SET {', '.join(fields)} WHERE user_id = %s RETURNING user_id",
            tuple(params),
        )
        row = await cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
    updated = await _fetch_user_by_id(user_id)
    return _public_user(updated)


async def record_login(user_id: int):
    async with await _get_conn() as con:
        await con.execute(
            "UPDATE app_user SET last_login = %s, updated_at = %s WHERE user_id = %s",
            (_now(), _now(), user_id),
        )

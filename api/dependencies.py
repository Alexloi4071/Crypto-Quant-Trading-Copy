"""
API Dependencies Module
依賴注入函數，用於 FastAPI 路由
"""

from typing import Optional, Dict, Any, Tuple
from fastapi import Request, HTTPException, Query, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

from api.config import api_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# HTTP Bearer 安全方案
security = HTTPBearer(auto_error=False)

# ============ 系統組件依賴 ============

def get_trading_system(request: Request):
    """獲取交易系統實例（必須存在）"""
    trading_system = getattr(request.app.state, 'trading_system', None)
    if trading_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="交易系統未啟動"
        )
    return trading_system

def get_trading_system_optional(request: Request):
    """獲取交易系統實例（可選）"""
    return getattr(request.app.state, 'trading_system', None)

def get_data_manager(request: Request):
    """獲取數據管理器實例"""
    data_manager = getattr(request.app.state, 'data_manager', None)
    if data_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="數據管理器未初始化"
        )
    return data_manager

def get_system_monitor(request: Request):
    """獲取系統監控實例"""
    system_monitor = getattr(request.app.state, 'system_monitor', None)
    if system_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="系統監控未初始化"
        )
    return system_monitor

def get_alerting_system(request: Request):
    """獲取告警系統實例"""
    alerting_system = getattr(request.app.state, 'alerting_system', None)
    if alerting_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="告警系統未初始化"
        )
    return alerting_system

def get_notification_service(request: Request):
    """獲取通知服務實例"""
    notification_service = getattr(request.app.state, 'notification_service', None)
    if notification_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="通知服務未初始化"
        )
    return notification_service

def get_health_checker(request: Request):
    """獲取健康檢查器實例"""
    health_checker = getattr(request.app.state, 'health_checker', None)
    if health_checker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="健康檢查器未初始化"
        )
    return health_checker

def get_db_manager(request: Request):
    """獲取數據庫管理器實例"""
    db_manager = getattr(request.app.state, 'db_manager', None)
    if db_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="數據庫管理器未初始化"
        )
    return db_manager

# ============ 認證與授權 ============

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """創建訪問令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=api_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, api_config.SECRET_KEY, algorithm=api_config.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """驗證令牌"""
    try:
        payload = jwt.decode(token, api_config.SECRET_KEY, algorithms=[api_config.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌已過期"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的令牌"
        )

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """獲取當前用戶（如果啟用認證）"""
    # 如果未啟用認證，返回默認用戶
    if not api_config.ENABLE_AUTH:
        return {
            "user_id": "default",
            "username": "default_user",
            "role": "admin",
            "permissions": ["all"]
        }
    
    # 啟用認證時驗證令牌
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少認證令牌"
        )
    
    token = credentials.credentials
    payload = verify_token(token)
    
    return {
        "user_id": payload.get("sub"),
        "username": payload.get("username"),
        "role": payload.get("role", "viewer"),
        "permissions": payload.get("permissions", [])
    }

def require_permission(permission: str):
    """要求特定權限的依賴"""
    async def permission_checker(current_user: Dict = Depends(get_current_user)):
        # 管理員擁有所有權限
        if current_user.get("role") == "admin":
            return current_user
        
        # 檢查用戶權限
        user_permissions = current_user.get("permissions", [])
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少權限: {permission}"
            )
        
        return current_user
    
    return permission_checker

# ============ 查詢參數依賴 ============

def get_pagination(
    page: int = Query(1, ge=1, description="頁碼"),
    page_size: int = Query(20, ge=1, le=100, description="每頁數量")
) -> Tuple[int, int]:
    """獲取分頁參數"""
    offset = (page - 1) * page_size
    return offset, page_size

def get_query_params(
    symbol: Optional[str] = Query(None, description="交易對"),
    timeframe: Optional[str] = Query(None, description="時間框架"),
    start_date: Optional[str] = Query(None, description="開始日期"),
    end_date: Optional[str] = Query(None, description="結束日期")
) -> Dict[str, Any]:
    """獲取通用查詢參數"""
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date
    }

# ============ 響應格式化 ============

def create_response(
    data: Any,
    message: str = "成功",
    code: int = 200
) -> Dict[str, Any]:
    """創建標準響應格式"""
    return {
        "success": True,
        "code": code,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }

def create_error_response(
    message: str,
    code: int = 500,
    detail: Optional[str] = None
) -> Dict[str, Any]:
    """創建錯誤響應格式"""
    response = {
        "success": False,
        "code": code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if detail:
        response["detail"] = detail
    
    return response

# ============ 請求驗證 ============

def validate_symbol(symbol: str) -> str:
    """驗證交易對格式"""
    symbol = symbol.upper()
    if not symbol.endswith('USDT'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="交易對必須以 USDT 結尾"
        )
    return symbol

def validate_timeframe(timeframe: str) -> str:
    """驗證時間框架"""
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"無效的時間框架，支持: {', '.join(valid_timeframes)}"
        )
    return timeframe

def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
    """驗證日期範圍"""
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        if start >= end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="開始日期必須早於結束日期"
            )
        
        return start, end
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="日期格式錯誤，應為 ISO 格式 (YYYY-MM-DD)"
        )

# ============ 錯誤處理 ============

def handle_service_error(e: Exception, service_name: str):
    """處理服務錯誤"""
    logger.error(f"{service_name} 錯誤: {str(e)}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{service_name} 服務錯誤: {str(e)}"
    )


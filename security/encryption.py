"""
Encryption Module
数据加密模块，提供多种加密算法和安全数据处理
支持对称加密、非对称加密和数字签名
"""

import os
import base64
import secrets
import hashlib
from typing import Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    ChaCha20_Poly1305 = "chacha20_poly1305"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"

class HashAlgorithm(Enum):
    """哈希算法枚举"""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    BLAKE2b = "blake2b"
    ARGON2 = "argon2"

@dataclass

class EncryptionKey:
    """加密密钥"""
    key_id: str
    key_data: bytes
    algorithm: EncryptionAlgorithm
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

@dataclass

class EncryptedData:
    """加密数据结构"""
    algorithm: EncryptionAlgorithm
    ciphertext: bytes
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None  # 用于认证加密
    key_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'algorithm': self.algorithm.value,
            'ciphertext': base64.b64encode(self.ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(self.nonce).decode('utf-8') if self.nonce else None,
            'tag': base64.b64encode(self.tag).decode('utf-8') if self.tag else None,
            'key_id': self.key_id,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod

    def from_dict(cls, data: dict) -> 'EncryptedData':
        """从字典创建"""
        return cls(
            algorithm=EncryptionAlgorithm(data['algorithm']),
            ciphertext=base64.b64decode(data['ciphertext']),
            nonce=base64.b64decode(data['nonce']) if data.get('nonce') else None,
            tag=base64.b64decode(data['tag']) if data.get('tag') else None,
            key_id=data.get('key_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else datetime.now()
        )

class AESEncryption:
    """AES加密实现"""

    def __init__(self):
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.primitives import padding
            from cryptography.hazmat.backends import default_backend

            self.Cipher = Cipher
            self.algorithms = algorithms
            self.modes = modes
            self.padding = padding
            self.backend = default_backend()

        except ImportError:
            logger.error("需要安装cryptography库: pip install cryptography")
            raise

    def generate_key(self) -> bytes:
        """生成AES密钥"""
        return os.urandom(32)  # 256位密钥

    def encrypt_gcm(self, plaintext: bytes, key: bytes,
                   associated_data: bytes = None) -> EncryptedData:
        """AES-GCM加密"""
        nonce = os.urandom(12)  # 96位nonce for GCM

        cipher = self.Cipher(
            self.algorithms.AES(key),
            self.modes.GCM(nonce),
            backend=self.backend
        )

        encryptor = cipher.encryptor()

        if associated_data:
            encryptor.authenticate_additional_data(associated_data)

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return EncryptedData(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            ciphertext=ciphertext,
            nonce=nonce,
            tag=encryptor.tag
        )

    def decrypt_gcm(self, encrypted_data: EncryptedData, key: bytes,
                   associated_data: bytes = None) -> bytes:
        """AES-GCM解密"""
        cipher = self.Cipher(
            self.algorithms.AES(key),
            self.modes.GCM(encrypted_data.nonce, encrypted_data.tag),
            backend=self.backend
        )

        decryptor = cipher.decryptor()

        if associated_data:
            decryptor.authenticate_additional_data(associated_data)

        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

    def encrypt_cbc(self, plaintext: bytes, key: bytes) -> EncryptedData:
        """AES-CBC加密"""
        # PKCS7填充
        padder = self.padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        iv = os.urandom(16)  # 128位IV for CBC

        cipher = self.Cipher(
            self.algorithms.AES(key),
            self.modes.CBC(iv),
            backend=self.backend
        )

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return EncryptedData(
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            ciphertext=ciphertext,
            nonce=iv  # 在CBC模式下使用nonce字段存储IV
        )

    def decrypt_cbc(self, encrypted_data: EncryptedData, key: bytes) -> bytes:
        """AES-CBC解密"""
        cipher = self.Cipher(
            self.algorithms.AES(key),
            self.modes.CBC(encrypted_data.nonce),
            backend=self.backend
        )

        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

        # 移除PKCS7填充
        unpadder = self.padding.PKCS7(128).unpadder()
        return unpadder.update(padded_plaintext) + unpadder.finalize()

class RSAEncryption:
    """RSA加密实现"""

    def __init__(self):
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.backends import default_backend

            self.rsa = rsa
            self.padding = padding
            self.hashes = hashes
            self.serialization = serialization
            self.backend = default_backend()

        except ImportError:
            logger.error("需要安装cryptography库: pip install cryptography")
            raise

    def generate_key_pair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """生成RSA密钥对"""
        private_key = self.rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )

        public_key = private_key.public_key()

        # 序列化私钥
        private_pem = private_key.private_bytes(
            encoding=self.serialization.Encoding.PEM,
            format=self.serialization.PrivateFormat.PKCS8,
            encryption_algorithm=self.serialization.NoEncryption()
        )

        # 序列化公钥
        public_pem = public_key.public_bytes(
            encoding=self.serialization.Encoding.PEM,
            format=self.serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    def encrypt(self, plaintext: bytes, public_key_pem: bytes) -> EncryptedData:
        """RSA加密"""
        public_key = self.serialization.load_pem_public_key(
            public_key_pem,
            backend=self.backend
        )

        ciphertext = public_key.encrypt(
            plaintext,
            self.padding.OAEP(
                mgf=self.padding.MGF1(algorithm=self.hashes.SHA256()),
                algorithm=self.hashes.SHA256(),
                label=None
            )
        )

        return EncryptedData(
            algorithm=EncryptionAlgorithm.RSA_2048,
            ciphertext=ciphertext
        )

    def decrypt(self, encrypted_data: EncryptedData, private_key_pem: bytes) -> bytes:
        """RSA解密"""
        private_key = self.serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )

        return private_key.decrypt(
            encrypted_data.ciphertext,
            self.padding.OAEP(
                mgf=self.padding.MGF1(algorithm=self.hashes.SHA256()),
                algorithm=self.hashes.SHA256(),
                label=None
            )
        )

    def sign(self, message: bytes, private_key_pem: bytes) -> bytes:
        """RSA数字签名"""
        private_key = self.serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )

        return private_key.sign(
            message,
            self.padding.PSS(
                mgf=self.padding.MGF1(self.hashes.SHA256()),
                salt_length=self.padding.PSS.MAX_LENGTH
            ),
            self.hashes.SHA256()
        )

    def verify_signature(self, message: bytes, signature: bytes,
                        public_key_pem: bytes) -> bool:
        """验证RSA数字签名"""
        try:
            public_key = self.serialization.load_pem_public_key(
                public_key_pem,
                backend=self.backend
            )

            public_key.verify(
                signature,
                message,
                self.padding.PSS(
                    mgf=self.padding.MGF1(self.hashes.SHA256()),
                    salt_length=self.padding.PSS.MAX_LENGTH
                ),
                self.hashes.SHA256()
            )

            return True

        except Exception as e:
            logger.error(f"签名验证失败: {e}")
            return False

class HashManager:
    """哈希管理器"""

    def __init__(self):
        self.algorithms = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA384: hashlib.sha384,
            HashAlgorithm.SHA512: hashlib.sha512,
        }

        # 尝试导入Blake2
        try:
            self.algorithms[HashAlgorithm.BLAKE2b] = hashlib.blake2b
        except AttributeError:
            logger.warning("Blake2b不可用")

        # 尝试导入Argon2
        try:
            import argon2
            self.argon2 = argon2.PasswordHasher()
        except ImportError:
            logger.warning("Argon2不可用，请安装: pip install argon2-cffi")
            self.argon2 = None

    def hash_data(self, data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """哈希数据"""
        if algorithm == HashAlgorithm.ARGON2:
            if self.argon2:
                return self.argon2.hash(data.decode('utf-8') if isinstance(data, bytes) else data)
            else:
                raise ValueError("Argon2不可用")

        if algorithm not in self.algorithms:
            raise ValueError(f"不支持的哈希算法: {algorithm}")

        hash_func = self.algorithms[algorithm]
        return hash_func(data).hexdigest()

    def verify_hash(self, data: bytes, hash_value: str,
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """验证哈希"""
        if algorithm == HashAlgorithm.ARGON2:
            if self.argon2:
                try:
                    self.argon2.verify(hash_value, data.decode('utf-8') if isinstance(data, bytes) else data)
                    return True
                except:
                    return False
            else:
                return False

        computed_hash = self.hash_data(data, algorithm)
        return secrets.compare_digest(computed_hash, hash_value)

    def generate_salt(self, length: int = 32) -> bytes:
        """生成盐值"""
        return os.urandom(length)

    def hash_password(self, password: str, salt: bytes = None) -> Tuple[str, bytes]:
        """安全哈希密码"""
        if salt is None:
            salt = self.generate_salt()

        if self.argon2:
            # 使用Argon2
            hash_value = self.argon2.hash(password)
            return hash_value, salt
        else:
            # 使用PBKDF2
            try:
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.backends import default_backend

                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )

                key = kdf.derive(password.encode('utf-8'))
                return base64.b64encode(key).decode('utf-8'), salt

            except ImportError:
                # 回退到简单的盐值哈希
                salted_password = salt + password.encode('utf-8')
                hash_value = self.hash_data(salted_password, HashAlgorithm.SHA256)
                return hash_value, salt

class KeyManager:
    """密钥管理器"""

    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_derivation_cache = {}  # 缓存派生密钥

    def generate_key(self, key_id: str, algorithm: EncryptionAlgorithm,
                    expires_in_days: int = None) -> EncryptionKey:
        """生成新密钥"""
        if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            key_data = os.urandom(32)  # 256位
        elif algorithm == EncryptionAlgorithm.ChaCha20_Poly1305:
            key_data = os.urandom(32)  # 256位
        else:
            raise ValueError(f"不支持的对称密钥算法: {algorithm}")

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        encryption_key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            algorithm=algorithm,
            expires_at=expires_at
        )

        self.keys[key_id] = encryption_key

        logger.info(f"生成加密密钥: {key_id} ({algorithm.value})")
        return encryption_key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """获取密钥"""
        key = self.keys.get(key_id)

        if key and key.is_expired():
            logger.warning(f"密钥已过期: {key_id}")
            return None

        return key

    def derive_key(self, master_key: bytes, context: str,
                  length: int = 32) -> bytes:
        """从主密钥派生子密钥"""
        # 使用HKDF派生密钥
        cache_key = f"{master_key.hex()}:{context}:{length}"

        if cache_key in self.key_derivation_cache:
            return self.key_derivation_cache[cache_key]

        try:
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=length,
                salt=None,
                info=context.encode('utf-8'),
                backend=default_backend()
            )

            derived_key = hkdf.derive(master_key)

            # 缓存结果
            self.key_derivation_cache[cache_key] = derived_key

            return derived_key

        except ImportError:
            # 简单的派生方法
            import hmac

            derived_key = hmac.new(
                master_key,
                context.encode('utf-8'),
                hashlib.sha256
            ).digest()[:length]

            self.key_derivation_cache[cache_key] = derived_key
            return derived_key

    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """轮换密钥"""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None

        # 生成新密钥
        new_key = self.generate_key(
            f"{key_id}_rotated_{int(datetime.now().timestamp())}",
            old_key.algorithm
        )

        # 保留旧密钥用于解密旧数据
        old_key.metadata['rotated_at'] = datetime.now().isoformat()
        old_key.metadata['successor'] = new_key.key_id

        logger.info(f"轮换密钥: {key_id} -> {new_key.key_id}")
        return new_key

    def cleanup_expired_keys(self):
        """清理过期密钥"""
        expired_keys = []

        for key_id, key in self.keys.items():
            if key.is_expired():
                expired_keys.append(key_id)

        for key_id in expired_keys:
            del self.keys[key_id]

        if expired_keys:
            logger.info(f"清理 {len(expired_keys)} 个过期密钥")

class EncryptionManager:
    """加密管理器主类"""

    def __init__(self):
        self.aes_encryption = AESEncryption()
        self.rsa_encryption = RSAEncryption()
        self.hash_manager = HashManager()
        self.key_manager = KeyManager()

        # 统计信息
        self.stats = {
            'encryptions_performed': 0,
            'decryptions_performed': 0,
            'keys_generated': 0,
            'hash_operations': 0,
            'signature_operations': 0
        }

        logger.info("加密管理器初始化完成")

    def encrypt_data(self, plaintext: Union[str, bytes], key_id: str = None,
                    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
                    associated_data: bytes = None) -> EncryptedData:
        """加密数据"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # 获取或生成密钥
        if key_id:
            encryption_key = self.key_manager.get_key(key_id)
            if not encryption_key:
                raise ValueError(f"密钥不存在或已过期: {key_id}")
            key_data = encryption_key.key_data
        else:
            # 生成临时密钥
            key_id = f"temp_{int(datetime.now().timestamp())}"
            encryption_key = self.key_manager.generate_key(key_id, algorithm)
            key_data = encryption_key.key_data

        # 执行加密
        if algorithm == EncryptionAlgorithm.AES_256_GCM:
            encrypted_data = self.aes_encryption.encrypt_gcm(plaintext, key_data, associated_data)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            encrypted_data = self.aes_encryption.encrypt_cbc(plaintext, key_data)
        else:
            raise ValueError(f"不支持的加密算法: {algorithm}")

        encrypted_data.key_id = key_id
        self.stats['encryptions_performed'] += 1

        return encrypted_data

    def decrypt_data(self, encrypted_data: EncryptedData,
                    associated_data: bytes = None) -> bytes:
        """解密数据"""
        # 获取密钥
        if not encrypted_data.key_id:
            raise ValueError("缺少密钥ID")

        encryption_key = self.key_manager.get_key(encrypted_data.key_id)
        if not encryption_key:
            raise ValueError(f"密钥不存在或已过期: {encrypted_data.key_id}")

        # 执行解密
        if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
            plaintext = self.aes_encryption.decrypt_gcm(encrypted_data, encryption_key.key_data, associated_data)
        elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC:
            plaintext = self.aes_encryption.decrypt_cbc(encrypted_data, encryption_key.key_data)
        else:
            raise ValueError(f"不支持的解密算法: {encrypted_data.algorithm}")

        self.stats['decryptions_performed'] += 1
        return plaintext

    def encrypt_with_rsa(self, plaintext: Union[str, bytes], public_key_pem: bytes) -> EncryptedData:
        """使用RSA加密"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        encrypted_data = self.rsa_encryption.encrypt(plaintext, public_key_pem)
        self.stats['encryptions_performed'] += 1

        return encrypted_data

    def decrypt_with_rsa(self, encrypted_data: EncryptedData, private_key_pem: bytes) -> bytes:
        """使用RSA解密"""
        plaintext = self.rsa_encryption.decrypt(encrypted_data, private_key_pem)
        self.stats['decryptions_performed'] += 1

        return plaintext

    def hash_data(self, data: Union[str, bytes],
                 algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """哈希数据"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        hash_value = self.hash_manager.hash_data(data, algorithm)
        self.stats['hash_operations'] += 1

        return hash_value

    def verify_hash(self, data: Union[str, bytes], hash_value: str,
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """验证哈希"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        result = self.hash_manager.verify_hash(data, hash_value, algorithm)
        self.stats['hash_operations'] += 1

        return result

    def sign_data(self, data: Union[str, bytes], private_key_pem: bytes) -> str:
        """数字签名"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        signature = self.rsa_encryption.sign(data, private_key_pem)
        self.stats['signature_operations'] += 1

        return base64.b64encode(signature).decode('utf-8')

    def verify_signature(self, data: Union[str, bytes], signature: str,
                        public_key_pem: bytes) -> bool:
        """验证数字签名"""
        if isinstance(data, str):
            data = data.encode('utf-8')

        signature_bytes = base64.b64decode(signature)
        result = self.rsa_encryption.verify_signature(data, signature_bytes, public_key_pem)
        self.stats['signature_operations'] += 1

        return result

    def generate_rsa_key_pair(self, key_size: int = 2048) -> Tuple[str, str]:
        """生成RSA密钥对"""
        private_key_pem, public_key_pem = self.rsa_encryption.generate_key_pair(key_size)
        self.stats['keys_generated'] += 1

        return (
            private_key_pem.decode('utf-8'),
            public_key_pem.decode('utf-8')
        )

    def secure_compare(self, a: str, b: str) -> bool:
        """安全字符串比较（防止时序攻击）"""
        return secrets.compare_digest(a, b)

    def generate_secure_token(self, length: int = 32) -> str:
        """生成安全令牌"""
        return secrets.token_urlsafe(length)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats,
            'active_keys': len(self.key_manager.keys),
            'cached_derivations': len(self.key_manager.key_derivation_cache)
        }

# 全局实例
_encryption_manager_instance = None

def get_encryption_manager() -> EncryptionManager:
    """获取加密管理器实例"""
    global _encryption_manager_instance
    if _encryption_manager_instance is None:
        _encryption_manager_instance = EncryptionManager()
    return _encryption_manager_instance

# 便利函数

def encrypt_sensitive_data(data: str, key_id: str = None) -> dict:
    """加密敏感数据的便利函数"""
    manager = get_encryption_manager()
    encrypted_data = manager.encrypt_data(data, key_id)
    return encrypted_data.to_dict()

def decrypt_sensitive_data(encrypted_dict: dict) -> str:
    """解密敏感数据的便利函数"""
    manager = get_encryption_manager()
    encrypted_data = EncryptedData.from_dict(encrypted_dict)
    plaintext = manager.decrypt_data(encrypted_data)
    return plaintext.decode('utf-8')

def hash_password(password: str) -> Tuple[str, str]:
    """密码哈希便利函数"""
    manager = get_encryption_manager()
    hash_value, salt = manager.hash_manager.hash_password(password)
    return hash_value, base64.b64encode(salt).decode('utf-8')

def verify_password(password: str, hash_value: str, salt_b64: str) -> bool:
    """密码验证便利函数"""
    manager = get_encryption_manager()
    salt = base64.b64decode(salt_b64)

    if manager.hash_manager.argon2:
        # Argon2验证
        try:
            manager.hash_manager.argon2.verify(hash_value, password)
            return True
        except:
            return False
    else:
        # PBKDF2或简单盐值验证
        computed_hash, _ = manager.hash_manager.hash_password(password, salt)
        return manager.secure_compare(hash_value, computed_hash)

"""
Cloud Integration
云服务集成系统，支持主流云平台的资源管理和服务调用
实现AWS、Azure、Google Cloud等云服务的统一接口
"""

import asyncio
import aiohttp
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.compute import ComputeManagementClient
from google.cloud import storage, compute_v1
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class CloudProvider(Enum):
    """云服务提供商"""
    AWS = "aws"
    AZURE = "azure"
    GOOGLE_CLOUD = "google_cloud"
    ALIBABA_CLOUD = "alibaba_cloud"
    TENCENT_CLOUD = "tencent_cloud"

class ServiceType(Enum):
    """服务类型"""
    COMPUTE = "compute"           # 计算服务
    STORAGE = "storage"           # 存储服务
    DATABASE = "database"         # 数据库服务
    NETWORK = "network"           # 网络服务
    MACHINE_LEARNING = "ml"       # 机器学习服务
    ANALYTICS = "analytics"       # 分析服务
    SERVERLESS = "serverless"     # 无服务器
    CONTAINER = "container"       # 容器服务

class ResourceStatus(Enum):
    """资源状态"""
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    DELETING = "deleting"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass

class CloudResource:
    """云资源"""
    resource_id: str
    resource_name: str
    provider: CloudProvider
    service_type: ServiceType

    # 资源配置
    region: str
    instance_type: Optional[str] = None
    size: Optional[str] = None

    # 状态信息
    status: ResourceStatus = ResourceStatus.UNKNOWN
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # 成本信息
    hourly_cost: float = 0.0
    total_cost: float = 0.0

    # 元数据
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'resource_id': self.resource_id,
            'resource_name': self.resource_name,
            'provider': self.provider.value,
            'service_type': self.service_type.value,
            'region': self.region,
            'instance_type': self.instance_type,
            'size': self.size,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'cost': {
                'hourly_cost': self.hourly_cost,
                'total_cost': self.total_cost
            },
            'tags': self.tags,
            'metadata': self.metadata
        }

@dataclass

class DeploymentConfig:
    """部署配置"""
    name: str
    provider: CloudProvider
    service_type: ServiceType

    # 基础配置
    region: str
    instance_type: str

    # 网络配置
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)

    # 存储配置
    storage_size: int = 100  # GB
    storage_type: str = "gp2"

    # 高可用配置
    availability_zones: List[str] = field(default_factory=list)
    min_instances: int = 1
    max_instances: int = 1

    # 标签和元数据
    tags: Dict[str, str] = field(default_factory=dict)
    user_data: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'provider': self.provider.value,
            'service_type': self.service_type.value,
            'region': self.region,
            'instance_type': self.instance_type,
            'network': {
                'vpc_id': self.vpc_id,
                'subnet_id': self.subnet_id,
                'security_groups': self.security_groups
            },
            'storage': {
                'size': self.storage_size,
                'type': self.storage_type
            },
            'scaling': {
                'availability_zones': self.availability_zones,
                'min_instances': self.min_instances,
                'max_instances': self.max_instances
            },
            'tags': self.tags,
            'user_data': self.user_data
        }

class CloudProviderInterface:
    """云服务提供商接口基类"""

    def __init__(self, provider: CloudProvider):
        self.provider = provider
        self.authenticated = False

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """认证"""
        raise NotImplementedError

    async def create_resource(self, config: DeploymentConfig) -> CloudResource:
        """创建资源"""
        raise NotImplementedError

    async def get_resource(self, resource_id: str) -> Optional[CloudResource]:
        """获取资源"""
        raise NotImplementedError

    async def list_resources(self, service_type: Optional[ServiceType] = None) -> List[CloudResource]:
        """列出资源"""
        raise NotImplementedError

    async def delete_resource(self, resource_id: str) -> bool:
        """删除资源"""
        raise NotImplementedError

    async def start_resource(self, resource_id: str) -> bool:
        """启动资源"""
        raise NotImplementedError

    async def stop_resource(self, resource_id: str) -> bool:
        """停止资源"""
        raise NotImplementedError

    async def get_cost_info(self, resource_id: str) -> Dict[str, float]:
        """获取成本信息"""
        raise NotImplementedError

class AWSProvider(CloudProviderInterface):
    """AWS云服务提供商"""

    def __init__(self):
        super().__init__(CloudProvider.AWS)
        self.ec2_client = None
        self.s3_client = None
        self.cloudwatch_client = None

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """AWS认证"""
        try:
            access_key = credentials.get('access_key_id')
            secret_key = credentials.get('secret_access_key')
            region = credentials.get('region', 'us-east-1')

            # 创建客户端
            self.ec2_client = boto3.client(
                'ec2',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )

            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )

            self.cloudwatch_client = boto3.client(
                'cloudwatch',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )

            # 测试连接
            self.ec2_client.describe_regions()
            self.authenticated = True

            logger.info("AWS认证成功")
            return True

        except Exception as e:
            logger.error(f"AWS认证失败: {e}")
            return False

    async def create_resource(self, config: DeploymentConfig) -> CloudResource:
        """创建AWS资源"""
        if not self.authenticated or not self.ec2_client:
            raise ValueError("AWS未认证")

        try:
            if config.service_type == ServiceType.COMPUTE:
                return await self._create_ec2_instance(config)
            elif config.service_type == ServiceType.STORAGE:
                return await self._create_s3_bucket(config)
            else:
                raise ValueError(f"不支持的服务类型: {config.service_type}")

        except Exception as e:
            logger.error(f"创建AWS资源失败: {e}")
            raise

    async def _create_ec2_instance(self, config: DeploymentConfig) -> CloudResource:
        """创建EC2实例"""

        # 构建启动参数
        launch_params = {
            'ImageId': self._get_ami_id(config.region),  # 获取适当的AMI ID
            'InstanceType': config.instance_type,
            'MinCount': config.min_instances,
            'MaxCount': config.max_instances,
            'KeyName': config.tags.get('key_pair'),  # SSH密钥对
            'TagSpecifications': [
                {
                    'ResourceType': 'instance',
                    'Tags': [{'Key': k, 'Value': v} for k, v in config.tags.items()]
                }
            ]
        }

        # 网络配置
        if config.subnet_id:
            launch_params['SubnetId'] = config.subnet_id

        if config.security_groups:
            launch_params['SecurityGroupIds'] = config.security_groups

        if config.user_data:
            launch_params['UserData'] = config.user_data

        # 存储配置
        launch_params['BlockDeviceMappings'] = [
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': config.storage_size,
                    'VolumeType': config.storage_type,
                    'DeleteOnTermination': True
                }
            }
        ]

        # 启动实例
        response = self.ec2_client.run_instances(**launch_params)

        instance = response['Instances'][0]
        instance_id = instance['InstanceId']

        # 创建资源对象
        resource = CloudResource(
            resource_id=instance_id,
            resource_name=config.name,
            provider=CloudProvider.AWS,
            service_type=ServiceType.COMPUTE,
            region=config.region,
            instance_type=config.instance_type,
            status=ResourceStatus.CREATING,
            tags=config.tags,
            metadata={
                'instance_data': instance,
                'ami_id': launch_params['ImageId']
            }
        )

        logger.info(f"创建AWS EC2实例: {instance_id}")
        return resource

    async def _create_s3_bucket(self, config: DeploymentConfig) -> CloudResource:
        """创建S3存储桶"""

        bucket_name = config.name.lower().replace('_', '-')

        try:
            if config.region != 'us-east-1':
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': config.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=bucket_name)

            # 设置标签
            if config.tags:
                tag_set = [{'Key': k, 'Value': v} for k, v in config.tags.items()]
                self.s3_client.put_bucket_tagging(
                    Bucket=bucket_name,
                    Tagging={'TagSet': tag_set}
                )

            resource = CloudResource(
                resource_id=bucket_name,
                resource_name=config.name,
                provider=CloudProvider.AWS,
                service_type=ServiceType.STORAGE,
                region=config.region,
                status=ResourceStatus.RUNNING,
                tags=config.tags,
                metadata={'bucket_name': bucket_name}
            )

            logger.info(f"创建AWS S3存储桶: {bucket_name}")
            return resource

        except Exception as e:
            logger.error(f"创建S3存储桶失败: {e}")
            raise

    def _get_ami_id(self, region: str) -> str:
        """获取适当的AMI ID"""
        # 简化实现，返回Amazon Linux 2的通用AMI ID
        ami_mappings = {
            'us-east-1': 'ami-0abcdef1234567890',
            'us-west-2': 'ami-0123456789abcdef0',
            'eu-west-1': 'ami-0fedcba0987654321'
        }
        return ami_mappings.get(region, 'ami-0abcdef1234567890')

    async def get_resource(self, resource_id: str) -> Optional[CloudResource]:
        """获取AWS资源"""
        if not self.authenticated:
            return None

        try:
            # 尝试作为EC2实例获取
            response = self.ec2_client.describe_instances(InstanceIds=[resource_id])

            if response['Reservations']:
                instance = response['Reservations'][0]['Instances'][0]

                # 转换状态
                aws_state = instance['State']['Name']
                status_mapping = {
                    'pending': ResourceStatus.CREATING,
                    'running': ResourceStatus.RUNNING,
                    'stopped': ResourceStatus.STOPPED,
                    'terminated': ResourceStatus.DELETING,
                    'stopping': ResourceStatus.STOPPED,
                    'shutting-down': ResourceStatus.DELETING
                }
                status = status_mapping.get(aws_state, ResourceStatus.UNKNOWN)

                # 获取标签
                tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}

                resource = CloudResource(
                    resource_id=resource_id,
                    resource_name=tags.get('Name', resource_id),
                    provider=CloudProvider.AWS,
                    service_type=ServiceType.COMPUTE,
                    region=instance['Placement']['AvailabilityZone'][:-1],
                    instance_type=instance['InstanceType'],
                    status=status,
                    tags=tags,
                    metadata={'instance_data': instance}
                )

                return resource

        except Exception as e:
            logger.error(f"获取AWS资源失败: {e}")

        return None

    async def list_resources(self, service_type: Optional[ServiceType] = None) -> List[CloudResource]:
        """列出AWS资源"""
        if not self.authenticated:
            return []

        resources = []

        try:
            if service_type is None or service_type == ServiceType.COMPUTE:
                # 列出EC2实例
                response = self.ec2_client.describe_instances()

                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        if instance['State']['Name'] != 'terminated':
                            resource = await self.get_resource(instance['InstanceId'])
                            if resource:
                                resources.append(resource)

            if service_type is None or service_type == ServiceType.STORAGE:
                # 列出S3存储桶
                response = self.s3_client.list_buckets()

                for bucket in response['Buckets']:
                    bucket_name = bucket['Name']

                    resource = CloudResource(
                        resource_id=bucket_name,
                        resource_name=bucket_name,
                        provider=CloudProvider.AWS,
                        service_type=ServiceType.STORAGE,
                        region='us-east-1',  # 默认区域
                        status=ResourceStatus.RUNNING,
                        created_at=bucket['CreationDate'].replace(tzinfo=None),
                        metadata={'bucket_name': bucket_name}
                    )

                    resources.append(resource)

        except Exception as e:
            logger.error(f"列出AWS资源失败: {e}")

        return resources

    async def delete_resource(self, resource_id: str) -> bool:
        """删除AWS资源"""
        if not self.authenticated:
            return False

        try:
            # 尝试作为EC2实例删除
            self.ec2_client.terminate_instances(InstanceIds=[resource_id])
            logger.info(f"删除AWS EC2实例: {resource_id}")
            return True

        except Exception:
            try:
                # 尝试作为S3存储桶删除
                self.s3_client.delete_bucket(Bucket=resource_id)
                logger.info(f"删除AWS S3存储桶: {resource_id}")
                return True
            except Exception as e:
                logger.error(f"删除AWS资源失败: {e}")
                return False

    async def start_resource(self, resource_id: str) -> bool:
        """启动AWS资源"""
        if not self.authenticated:
            return False

        try:
            self.ec2_client.start_instances(InstanceIds=[resource_id])
            logger.info(f"启动AWS EC2实例: {resource_id}")
            return True
        except Exception as e:
            logger.error(f"启动AWS资源失败: {e}")
            return False

    async def stop_resource(self, resource_id: str) -> bool:
        """停止AWS资源"""
        if not self.authenticated:
            return False

        try:
            self.ec2_client.stop_instances(InstanceIds=[resource_id])
            logger.info(f"停止AWS EC2实例: {resource_id}")
            return True
        except Exception as e:
            logger.error(f"停止AWS资源失败: {e}")
            return False

    async def get_cost_info(self, resource_id: str) -> Dict[str, float]:
        """获取AWS成本信息"""
        # 简化实现，实际应该调用AWS Cost Explorer API
        return {
            'hourly_cost': 0.10,  # 示例成本
            'daily_cost': 2.40,
            'monthly_cost': 72.00
        }

class AzureProvider(CloudProviderInterface):
    """Azure云服务提供商"""

    def __init__(self):
        super().__init__(CloudProvider.AZURE)
        self.credential = None
        self.resource_client = None
        self.compute_client = None
        self.subscription_id = None

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Azure认证"""
        try:
            self.subscription_id = credentials.get('subscription_id')

            # 使用默认凭证（支持多种认证方式）
            self.credential = DefaultAzureCredential()

            # 创建客户端
            self.resource_client = ResourceManagementClient(
                self.credential, self.subscription_id
            )

            self.compute_client = ComputeManagementClient(
                self.credential, self.subscription_id
            )

            # 测试连接
            list(self.resource_client.resource_groups.list())

            self.authenticated = True
            logger.info("Azure认证成功")
            return True

        except Exception as e:
            logger.error(f"Azure认证失败: {e}")
            return False

    async def create_resource(self, config: DeploymentConfig) -> CloudResource:
        """创建Azure资源"""
        if not self.authenticated:
            raise ValueError("Azure未认证")

        # 简化实现
        resource = CloudResource(
            resource_id=f"azure-{config.name}",
            resource_name=config.name,
            provider=CloudProvider.AZURE,
            service_type=config.service_type,
            region=config.region,
            instance_type=config.instance_type,
            status=ResourceStatus.CREATING,
            tags=config.tags
        )

        logger.info(f"创建Azure资源: {resource.resource_id}")
        return resource

    async def get_resource(self, resource_id: str) -> Optional[CloudResource]:
        """获取Azure资源"""
        # 简化实现
        return None

    async def list_resources(self, service_type: Optional[ServiceType] = None) -> List[CloudResource]:
        """列出Azure资源"""
        # 简化实现
        return []

    async def delete_resource(self, resource_id: str) -> bool:
        """删除Azure资源"""
        # 简化实现
        return True

    async def start_resource(self, resource_id: str) -> bool:
        """启动Azure资源"""
        # 简化实现
        return True

    async def stop_resource(self, resource_id: str) -> bool:
        """停止Azure资源"""
        # 简化实现
        return True

    async def get_cost_info(self, resource_id: str) -> Dict[str, float]:
        """获取Azure成本信息"""
        return {
            'hourly_cost': 0.12,
            'daily_cost': 2.88,
            'monthly_cost': 86.40
        }

class GoogleCloudProvider(CloudProviderInterface):
    """Google Cloud云服务提供商"""

    def __init__(self):
        super().__init__(CloudProvider.GOOGLE_CLOUD)
        self.compute_client = None
        self.storage_client = None
        self.project_id = None

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Google Cloud认证"""
        try:
            self.project_id = credentials.get('project_id')

            # 创建客户端（假设已设置GOOGLE_APPLICATION_CREDENTIALS环境变量）
            self.compute_client = compute_v1.InstancesClient()
            self.storage_client = storage.Client(project=self.project_id)

            # 测试连接
            list(self.storage_client.list_buckets())

            self.authenticated = True
            logger.info("Google Cloud认证成功")
            return True

        except Exception as e:
            logger.error(f"Google Cloud认证失败: {e}")
            return False

    async def create_resource(self, config: DeploymentConfig) -> CloudResource:
        """创建Google Cloud资源"""
        if not self.authenticated:
            raise ValueError("Google Cloud未认证")

        # 简化实现
        resource = CloudResource(
            resource_id=f"gcp-{config.name}",
            resource_name=config.name,
            provider=CloudProvider.GOOGLE_CLOUD,
            service_type=config.service_type,
            region=config.region,
            instance_type=config.instance_type,
            status=ResourceStatus.CREATING,
            tags=config.tags
        )

        logger.info(f"创建Google Cloud资源: {resource.resource_id}")
        return resource

    async def get_resource(self, resource_id: str) -> Optional[CloudResource]:
        """获取Google Cloud资源"""
        # 简化实现
        return None

    async def list_resources(self, service_type: Optional[ServiceType] = None) -> List[CloudResource]:
        """列出Google Cloud资源"""
        # 简化实现
        return []

    async def delete_resource(self, resource_id: str) -> bool:
        """删除Google Cloud资源"""
        # 简化实现
        return True

    async def start_resource(self, resource_id: str) -> bool:
        """启动Google Cloud资源"""
        # 简化实现
        return True

    async def stop_resource(self, resource_id: str) -> bool:
        """停止Google Cloud资源"""
        # 简化实现
        return True

    async def get_cost_info(self, resource_id: str) -> Dict[str, float]:
        """获取Google Cloud成本信息"""
        return {
            'hourly_cost': 0.08,
            'daily_cost': 1.92,
            'monthly_cost': 57.60
        }

class CloudIntegrationSystem:
    """云服务集成系统主类"""

    def __init__(self):
        self.providers = {}
        self.resources = {}  # resource_id -> CloudResource

        # 注册云服务提供商
        self._register_providers()

        # 统计信息
        self.stats = {
            'total_resources': 0,
            'resources_by_provider': defaultdict(int),
            'resources_by_type': defaultdict(int),
            'total_cost': 0.0,
            'monthly_cost': 0.0
        }

        logger.info("云服务集成系统初始化完成")

    def _register_providers(self):
        """注册云服务提供商"""
        self.providers[CloudProvider.AWS] = AWSProvider()
        self.providers[CloudProvider.AZURE] = AzureProvider()
        self.providers[CloudProvider.GOOGLE_CLOUD] = GoogleCloudProvider()

    async def authenticate_provider(self, provider: CloudProvider,
                                  credentials: Dict[str, str]) -> bool:
        """认证云服务提供商"""
        if provider not in self.providers:
            logger.error(f"不支持的云服务提供商: {provider}")
            return False

        return await self.providers[provider].authenticate(credentials)

    async def deploy_resource(self, config: DeploymentConfig) -> CloudResource:
        """部署云资源"""
        provider = self.providers.get(config.provider)

        if not provider:
            raise ValueError(f"不支持的云服务提供商: {config.provider}")

        if not provider.authenticated:
            raise ValueError(f"云服务提供商未认证: {config.provider}")

        # 创建资源
        resource = await provider.create_resource(config)

        # 记录资源
        self.resources[resource.resource_id] = resource
        self._update_stats()

        logger.info(f"部署云资源成功: {resource.resource_id}")
        return resource

    async def get_resource_info(self, resource_id: str) -> Optional[CloudResource]:
        """获取资源信息"""
        # 从缓存获取
        if resource_id in self.resources:
            resource = self.resources[resource_id]

            # 从云服务提供商更新状态
            provider = self.providers.get(resource.provider)
            if provider and provider.authenticated:
                updated_resource = await provider.get_resource(resource_id)
                if updated_resource:
                    self.resources[resource_id] = updated_resource
                    return updated_resource

            return resource

        # 尝试从所有已认证的提供商获取
        for provider in self.providers.values():
            if provider.authenticated:
                resource = await provider.get_resource(resource_id)
                if resource:
                    self.resources[resource_id] = resource
                    self._update_stats()
                    return resource

        return None

    async def list_all_resources(self, provider: Optional[CloudProvider] = None,
                               service_type: Optional[ServiceType] = None) -> List[CloudResource]:
        """列出所有资源"""
        all_resources = []

        providers_to_check = [provider] if provider else self.providers.keys()

        for provider_key in providers_to_check:
            provider_obj = self.providers[provider_key]
            if provider_obj.authenticated:
                try:
                    resources = await provider_obj.list_resources(service_type)

                    # 更新缓存
                    for resource in resources:
                        self.resources[resource.resource_id] = resource

                    all_resources.extend(resources)

                except Exception as e:
                    logger.error(f"列出 {provider_key.value} 资源失败: {e}")

        self._update_stats()
        return all_resources

    async def delete_resource(self, resource_id: str) -> bool:
        """删除资源"""
        resource = await self.get_resource_info(resource_id)

        if not resource:
            logger.error(f"资源不存在: {resource_id}")
            return False

        provider = self.providers.get(resource.provider)
        if not provider or not provider.authenticated:
            logger.error(f"云服务提供商未认证: {resource.provider}")
            return False

        # 删除资源
        success = await provider.delete_resource(resource_id)

        if success:
            # 从缓存中移除
            if resource_id in self.resources:
                del self.resources[resource_id]

            self._update_stats()
            logger.info(f"删除资源成功: {resource_id}")

        return success

    async def start_resource(self, resource_id: str) -> bool:
        """启动资源"""
        resource = await self.get_resource_info(resource_id)

        if not resource:
            return False

        provider = self.providers.get(resource.provider)
        if not provider or not provider.authenticated:
            return False

        return await provider.start_resource(resource_id)

    async def stop_resource(self, resource_id: str) -> bool:
        """停止资源"""
        resource = await self.get_resource_info(resource_id)

        if not resource:
            return False

        provider = self.providers.get(resource.provider)
        if not provider or not provider.authenticated:
            return False

        return await provider.stop_resource(resource_id)

    async def get_cost_analysis(self) -> Dict[str, Any]:
        """获取成本分析"""
        total_cost = 0.0
        cost_by_provider = defaultdict(float)
        cost_by_service = defaultdict(float)

        for resource in self.resources.values():
            provider = self.providers.get(resource.provider)
            if provider and provider.authenticated:
                try:
                    cost_info = await provider.get_cost_info(resource.resource_id)
                    monthly_cost = cost_info.get('monthly_cost', 0.0)

                    total_cost += monthly_cost
                    cost_by_provider[resource.provider.value] += monthly_cost
                    cost_by_service[resource.service_type.value] += monthly_cost

                    # 更新资源成本信息
                    resource.hourly_cost = cost_info.get('hourly_cost', 0.0)
                    resource.total_cost = monthly_cost

                except Exception as e:
                    logger.error(f"获取资源 {resource.resource_id} 成本失败: {e}")

        return {
            'total_monthly_cost': total_cost,
            'cost_by_provider': dict(cost_by_provider),
            'cost_by_service': dict(cost_by_service),
            'total_resources': len(self.resources),
            'cost_per_resource': total_cost / len(self.resources) if self.resources else 0
        }

    def _update_stats(self):
        """更新统计信息"""
        self.stats['total_resources'] = len(self.resources)
        self.stats['resources_by_provider'].clear()
        self.stats['resources_by_type'].clear()

        total_cost = 0.0

        for resource in self.resources.values():
            self.stats['resources_by_provider'][resource.provider.value] += 1
            self.stats['resources_by_type'][resource.service_type.value] += 1
            total_cost += resource.total_cost

        self.stats['total_cost'] = total_cost
        self.stats['monthly_cost'] = total_cost

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        authenticated_providers = [
            provider.value for provider, obj in self.providers.items()
            if obj.authenticated
        ]

        return {
            'stats': dict(self.stats),
            'authenticated_providers': authenticated_providers,
            'total_providers': len(self.providers),
            'resources_summary': [
                resource.to_dict() for resource in list(self.resources.values())[:10]
            ]  # 只返回前10个资源的摘要
        }

# 便利函数

def create_deployment_configs() -> List[DeploymentConfig]:
    """创建示例部署配置"""
    configs = [
        DeploymentConfig(
            name="trading-server",
            provider=CloudProvider.AWS,
            service_type=ServiceType.COMPUTE,
            region="us-east-1",
            instance_type="t3.medium",
            storage_size=100,
            tags={"Environment": "production", "Application": "trading"}
        ),
        DeploymentConfig(
            name="data-storage",
            provider=CloudProvider.AWS,
            service_type=ServiceType.STORAGE,
            region="us-east-1",
            instance_type="",
            tags={"Environment": "production", "Application": "data-lake"}
        ),
        DeploymentConfig(
            name="ml-training",
            provider=CloudProvider.GOOGLE_CLOUD,
            service_type=ServiceType.COMPUTE,
            region="us-central1",
            instance_type="n1-standard-4",
            storage_size=500,
            tags={"Environment": "development", "Application": "ml-training"}
        )
    ]

    return configs

# 全局实例
_cloud_integration_system_instance = None

def get_cloud_integration_system() -> CloudIntegrationSystem:
    """获取云服务集成系统实例"""
    global _cloud_integration_system_instance
    if _cloud_integration_system_instance is None:
        _cloud_integration_system_instance = CloudIntegrationSystem()
    return _cloud_integration_system_instance

"""工具配置管理 - 企业级最佳实践

支持从配置文件、环境变量等加载工具配置。
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

# 可选导入yaml
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyYAML未安装，YAML配置文件功能不可用。请运行: pip install pyyaml")

logger = logging.getLogger(__name__)


@dataclass
class ToolConfig:
    """工具配置"""
    name: str
    category: str = "custom"
    permission: str = "public"
    allowed_agents: List[str] = None
    tags: List[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    rate_limit: Optional[int] = None
    timeout: Optional[float] = None
    requires_auth: bool = False
    cost_per_call: float = 0.0
    enabled: bool = True
    custom_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.allowed_agents is None:
            self.allowed_agents = []
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}


class ToolConfigManager:
    """工具配置管理器"""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> List[ToolConfig]:
        """
        从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            工具配置列表
        """
        if not YAML_AVAILABLE:
            logger.error("PyYAML未安装，无法加载YAML配置文件。请运行: pip install pyyaml")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            configs = []
            for tool_data in data.get('tools', []):
                configs.append(ToolConfig(**tool_data))
            
            logger.info(f"从 {file_path} 加载了 {len(configs)} 个工具配置")
            return configs
        except Exception as e:
            logger.error(f"加载YAML配置失败: {e}")
            return []
    
    @staticmethod
    def load_from_json(file_path: str) -> List[ToolConfig]:
        """
        从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            工具配置列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            configs = []
            for tool_data in data.get('tools', []):
                configs.append(ToolConfig(**tool_data))
            
            logger.info(f"从 {file_path} 加载了 {len(configs)} 个工具配置")
            return configs
        except Exception as e:
            logger.error(f"加载JSON配置失败: {e}")
            return []
    
    @staticmethod
    def save_to_yaml(configs: List[ToolConfig], file_path: str):
        """
        保存配置到YAML文件
        
        Args:
            configs: 工具配置列表
            file_path: 输出文件路径
        """
        if not YAML_AVAILABLE:
            logger.error("PyYAML未安装，无法保存YAML配置文件。请运行: pip install pyyaml")
            return
        
        try:
            data = {
                "tools": [asdict(config) for config in configs]
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存YAML配置失败: {e}")
    
    @staticmethod
    def save_to_json(configs: List[ToolConfig], file_path: str):
        """
        保存配置到JSON文件
        
        Args:
            configs: 工具配置列表
            file_path: 输出文件路径
        """
        try:
            data = {
                "tools": [asdict(config) for config in configs]
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存JSON配置失败: {e}")


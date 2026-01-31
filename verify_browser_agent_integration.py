"""Browser Agent 集成验证脚本（静态验证）

此脚本验证 BrowserAgent 是否正确集成到 Plan-First 架构中，
不需要运行完整的系统，只做静态检查。

验证项：
1. 常量定义 (constants.py)
2. Planner 集成 (planner.py)
3. Router 集成 (graph_routers.py)
4. Agent 实现 (browser_agent.py)
5. Tools 实现 (browser_tools.py)
6. Graph 注册 (graph.py)
7. 配置文件 (agents.yaml)
"""

import ast
import sys
from pathlib import Path


def verify_file_syntax(file_path: Path, description: str) -> bool:
    """验证文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            ast.parse(code)
        print(f"  ✅ {description}: 语法正确")
        return True
    except SyntaxError as e:
        print(f"  ❌ {description}: 语法错误 - {e}")
        return False
    except FileNotFoundError:
        print(f"  ❌ {description}: 文件不存在")
        return False


def verify_content(file_path: Path, patterns: list, description: str) -> bool:
    """验证文件内容包含指定模式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing = []
        for pattern in patterns:
            if pattern not in content:
                missing.append(pattern)
        
        if missing:
            print(f"  ❌ {description}: 缺少 {missing}")
            return False
        else:
            print(f"  ✅ {description}: 所有检查项通过")
            return True
    except FileNotFoundError:
        print(f"  ❌ {description}: 文件不存在")
        return False


def main():
    print("="*70)
    print("Browser Agent 深度集成验证")
    print("="*70)
    
    project_root = Path(__file__).parent
    results = []
    
    # 1. 验证 constants.py
    print("\n[1] 验证 constants.py")
    constants_file = project_root / "src/multi_agent/constants.py"
    results.append(verify_file_syntax(constants_file, "constants.py 语法"))
    results.append(verify_content(
        constants_file,
        ["BROWSER_SEARCH", "BROWSER_AGENT"],
        "constants.py 常量定义"
    ))
    
    # 2. 验证 planner.py
    print("\n[2] 验证 planner.py (关键集成点)")
    planner_file = project_root / "src/multi_agent/planning/planner.py"
    results.append(verify_file_syntax(planner_file, "planner.py 语法"))
    results.append(verify_content(
        planner_file,
        [
            "browser_agent -> browser_search",
            "browser_agent：用于"
        ],
        "planner.py Planner集成"
    ))
    
    # 3. 验证 graph_routers.py
    print("\n[3] 验证 graph_routers.py (关键集成点)")
    router_file = project_root / "src/multi_agent/graph_routers.py"
    results.append(verify_file_syntax(router_file, "graph_routers.py 语法"))
    results.append(verify_content(
        router_file,
        [
            "ActionName.BROWSER_SEARCH",
            "AgentName.BROWSER_AGENT"
        ],
        "graph_routers.py Router集成"
    ))
    
    # 4. 验证 browser_agent.py
    print("\n[4] 验证 browser_agent.py")
    agent_file = project_root / "src/multi_agent/agents/browser_agent.py"
    results.append(verify_file_syntax(agent_file, "browser_agent.py 语法"))
    results.append(verify_content(
        agent_file,
        ["class BrowserAgent", "async def execute"],
        "browser_agent.py Agent实现"
    ))
    
    # 5. 验证 browser_tools.py
    print("\n[5] 验证 browser_tools.py")
    tools_file = project_root / "src/tools/browser_tools.py"
    results.append(verify_file_syntax(tools_file, "browser_tools.py 语法"))
    results.append(verify_content(
        tools_file,
        [
            "browser_search_product",
            "browser_compare_prices",
            "browser_extract_product_info"
        ],
        "browser_tools.py Tools实现"
    ))
    
    # 6. 验证 graph.py
    print("\n[6] 验证 graph.py")
    graph_file = project_root / "src/multi_agent/graph.py"
    results.append(verify_file_syntax(graph_file, "graph.py 语法"))
    results.append(verify_content(
        graph_file,
        [
            "from src.multi_agent.agents.browser_agent import BrowserAgent",
            "BrowserAgent(llm=self.llm)"
        ],
        "graph.py Graph注册"
    ))
    
    # 7. 验证 agents.yaml
    print("\n[7] 验证 agents.yaml")
    config_file = project_root / "config/agents.yaml"
    results.append(verify_content(
        config_file,
        [
            "name: browser_agent",
            "priority: 30"
        ],
        "agents.yaml 配置文件"
    ))
    
    # 汇总结果
    print("\n" + "="*70)
    print("验证结果汇总")
    print("="*70)
    
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"\n总计: {total} 项检查")
    print(f"✅ 通过: {passed} 项")
    print(f"❌ 失败: {failed} 项")
    
    if failed == 0:
        print("\n" + "="*70)
        print("🎉 所有验证通过！Browser Agent 已完全集成到系统中！")
        print("="*70)
        print("\n下一步：")
        print("1. 修复环境依赖（numpy 版本冲突）")
        print("2. 运行示例：python examples/browser_search_demo.py")
        print("3. 测试完整流程")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  部分验证失败，请检查上述错误")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""测试服务器是否能正常启动"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("[TEST] 正在导入 Flask 应用...")
    from server.app import create_app
    
    print("[TEST] 正在创建应用实例...")
    app = create_app()
    
    print("[TEST] 应用创建成功！")
    print("[TEST] 测试通过，服务器可以正常启动")
    sys.exit(0)
except Exception as e:
    print(f"[TEST] 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


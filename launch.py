#!/usr/bin/env python3
"""
Auto Audio 系统 - 最终简化启动脚本
Python 3.13.2 兼容版本

这是唯一需要的启动脚本，提供完整的语音识别和文本分析功能
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_logging():
    """设置日志"""
    log_dir = project_root / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "system.log")
        ]
    )

def check_dependencies():
    """检查依赖包"""
    print("检查系统依赖...")
    
    dependencies = []
    
    # 检查基础包
    try:
        import numpy as np
        dependencies.append("numpy ✓")
    except ImportError:
        dependencies.append("numpy ✗")
    
    # 安全地检查语音识别引擎（避免pkgutil错误）
    whisper_available = False
    try:
        # 尝试导入但捕获pkgutil错误
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is not None:
            dependencies.append("faster-whisper ✓")
            whisper_available = True
    except:
        pass
    
    if not whisper_available:
        try:
            import importlib.util
            if importlib.util.find_spec("whisper") is not None:
                dependencies.append("openai-whisper ✓")
                whisper_available = True
        except:
            pass
    
    if not whisper_available:
        try:
            import importlib.util
            if importlib.util.find_spec("speech_recognition") is not None:
                dependencies.append("SpeechRecognition ✓")
                whisper_available = True
        except:
            pass
    
    if not whisper_available:
        dependencies.append("语音识别引擎 ✗ (可选)")
    
    for dep in dependencies:
        print(f"  {dep}")
    
    return True

def start_educational_monitor():
    """启动教育语义监控模式"""
    try:
        from nlp.educational_analyzer import EducationalSemanticAnalyzer
        analyzer = EducationalSemanticAnalyzer()
        
        print("\\n" + "="*80)
        print("🔍 教育语义监控系统 - 课堂教学内容实时分析")
        print("="*80)
        print("📚 专业功能:")
        print("• 意识形态风险检测与预警")
        print("• 教学内容语义分析")
        print("• 前后逻辑一致性分析")
        print("• 实时风险评级与督学建议")
        print("• 输入 'help' 查看命令，'quit' 退出")
        print("• 输入 'summary' 查看分析摘要")
        print("="*80)
        
        while True:
            try:
                text = input("\\n🎓 请输入课堂教学内容 > ").strip()
                
                if not text:
                    continue
                    
                if text.lower() in ['quit', 'exit', '退出']:
                    print("\\n👋 退出教育监控系统")
                    break
                    
                if text.lower() in ['help', '帮助']:
                    print("\\n📖 命令说明:")
                    print("• 直接输入文本: 进行教育语义分析")
                    print("• summary: 查看24小时分析摘要")
                    print("• history: 查看分析历史")
                    print("• config: 查看当前配置")
                    print("• quit/exit: 退出系统")
                    continue
                    
                if text.lower() == 'summary':
                    summary = analyzer.get_analysis_summary(24)
                    print("\\n📊 24小时分析摘要:")
                    print(f"• 总分析次数: {summary.get('total_analyses', 0)}")
                    print(f"• 平均风险评分: {summary.get('average_risk_score', 0)}")
                    print(f"• 高风险检测次数: {summary.get('high_risk_count', 0)}")
                    if summary.get('frequent_risk_keywords'):
                        print("• 频繁风险关键词:", summary['frequent_risk_keywords'])
                    continue
                    
                if text.lower() == 'history':
                    if analyzer.analysis_history:
                        print(f"\\n📜 最近{min(5, len(analyzer.analysis_history))}次分析:")
                        for i, record in enumerate(analyzer.analysis_history[-5:], 1):
                            print(f"{i}. {record['timestamp'][:19]} - 风险等级: {record['risk_level']} - 评分: {record['risk_score']['total_score']}")
                    else:
                        print("\\n暂无分析历史")
                    continue
                    
                if text.lower() == 'config':
                    print("\\n⚙️ 系统配置:")
                    print(f"• 正面关键词数量: {len(analyzer.positive_educational_keywords)}")
                    print(f"• 风险关键词数量: {len(analyzer.risk_keywords)}")
                    print(f"• 风险模式数量: {len(analyzer.risk_patterns)}")
                    print(f"• 历史记录上限: {analyzer.max_history}")
                    continue
                
                # 进行教育语义分析
                print("\\n🔍 正在分析...")
                result = analyzer.analyze_educational_content(text)
                
                # 显示分析结果
                print("\\n" + "="*60)
                print("📋 分析结果")
                print("="*60)
                
                # 风险等级显示
                risk_level = result['risk_level']
                risk_emoji = {
                    'low': '🟢',
                    'medium': '🟡', 
                    'high': '🟠',
                    'critical': '🔴'
                }
                print(f"🎯 风险等级: {risk_emoji.get(risk_level, '⚪')} {risk_level.upper()}")
                print(f"📊 风险评分: {result['risk_score']['total_score']}/100")
                
                # 关键词分析
                if result['keyword_analysis']['risk_keywords']:
                    print(f"\\n⚠️ 检测到风险关键词:")
                    for kw in result['keyword_analysis']['risk_keywords'][:5]:
                        print(f"  • {kw['keyword']} (权重: {kw['weight']}, 出现: {kw['count']}次)")
                
                if result['keyword_analysis']['positive_keywords']:
                    print(f"\\n✅ 检测到正面关键词:")
                    for kw in result['keyword_analysis']['positive_keywords'][:3]:
                        print(f"  • {kw['keyword']} (出现: {kw['count']}次)")
                
                # 模式分析
                if result['pattern_analysis']['found_patterns']:
                    print(f"\\n🔍 检测到风险模式:")
                    for pattern in result['pattern_analysis']['found_patterns']:
                        print(f"  • {pattern['type']}: {pattern['description']}")
                
                # 逻辑一致性
                if result['logic_analysis']['inconsistencies']:
                    print(f"\\n🧠 逻辑一致性问题:")
                    for issue in result['logic_analysis']['inconsistencies']:
                        print(f"  • {issue}")
                else:
                    print(f"\\n🧠 逻辑一致性: {result['logic_analysis']['consistency_score']}/100")
                
                # 督学建议
                if result['recommendations']:
                    print(f"\\n💡 督学建议:")
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"  {i}. {rec}")
                
                print("="*60)
                
            except KeyboardInterrupt:
                print("\\n\\n👋 用户中断，退出监控")
                break
            except Exception as e:
                print(f"❌ 分析过程出错: {e}")
                continue
                
    except ImportError as e:
        print(f"❌ 教育分析器导入失败: {e}")
        print("切换到基础文本分析模式...")
        start_interactive_mode()
    except Exception as e:
        print(f"❌ 教育监控系统启动失败: {e}")
        start_interactive_mode()

def start_interactive_mode():
    """启动交互式模式"""
    try:
        from nlp.analyzer_independent import IndependentTextAnalyzer
        analyzer = IndependentTextAnalyzer()
        
        print("\\n" + "="*60)
        print("🎯 Auto Audio 文本分析器")
        print("="*60)
        print("功能说明:")
        print("• 支持中英文文本分析")
        print("• 提供词频统计、关键词提取")
        print("• 自动语言检测和情感分析")
        print("• 输入 'help' 查看命令，'quit' 退出")
        print("="*60)
        
        while True:
            try:
                text = input("\\n📝 请输入文本 > ").strip()
                
                if text.lower() in ['quit', 'q', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if text.lower() in ['help', '帮助', 'h']:
                    print("\\n📚 可用命令:")
                    print("  help/帮助 - 显示此帮助")
                    print("  quit/退出 - 退出程序")
                    print("  直接输入文本进行分析")
                    continue
                
                if not text:
                    print("⚠️ 请输入有效文本")
                    continue
                
                # 分析文本
                result = analyzer.analyze(text)
                
                print(f"\\n📊 分析结果:")
                print(f"   📏 字符数: {result['char_count']}")
                print(f"   🔤 词数: {result['word_count']}")
                print(f"   🌐 语言: {result['language']}")
                print(f"   😊 情感: {result['sentiment']}")
                
                if result['keywords']:
                    print(f"   🔑 关键词: {', '.join(result['keywords'][:5])}")
                
                if result['word_frequency']:
                    top_words = list(result['word_frequency'].keys())[:5]
                    print(f"   📈 高频词: {', '.join(top_words)}")
                
                # 显示分词结果
                if result['tokens']:
                    tokens_preview = result['tokens'][:8]
                    if len(result['tokens']) > 8:
                        tokens_preview.append("...")
                    print(f"   ✂️ 分词: {' | '.join(tokens_preview)}")
                
            except KeyboardInterrupt:
                print("\\n\\n👋 用户中断，退出程序")
                break
            except Exception as e:
                print(f"❌ 分析出错: {e}")
        
    except Exception as e:
        print(f"❌ 交互式模式启动失败: {e}")

def start_web_mode():
    """启动Web界面模式"""
    try:
        from web.app import app
        
        print("\\n🌐 启动Web界面...")
        print("Web界面地址: http://localhost:5000")
        print("按 Ctrl+C 停止服务")
        print("="*60)
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError:
        print("❌ Web模块不可用，使用交互式模式")
        start_interactive_mode()
    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")
        print("切换到交互式模式...")
        start_interactive_mode()

def main():
    """主函数"""
    setup_logging()
    
    print("="*60)
    print("🎵 Auto Audio 系统")
    print("Python 3.13.2 兼容版本")
    print("="*60)
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"项目路径: {project_root}")
    
    # 检查依赖
    check_dependencies()
      # 测试核心模块
    print("\\n测试核心模块...")
    try:
        from nlp.analyzer_independent import IndependentTextAnalyzer
        analyzer = IndependentTextAnalyzer()
        info = analyzer.get_analyzer_info()
        print(f"  ✓ 基础文本分析器: {info['description']}")
    except Exception as e:
        print(f"  ❌ 基础文本分析器加载失败: {e}")
        return 1
    
    try:
        from nlp.educational_analyzer import EducationalSemanticAnalyzer
        edu_analyzer = EducationalSemanticAnalyzer()
        print(f"  ✓ 教育语义监控分析器: 专业版")
    except Exception as e:
        print(f"  ⚠️ 教育语义监控分析器: 不可用 ({e})")
        print("  提示: 将使用基础文本分析功能")
      # 选择运行模式
    print("\\n选择运行模式:")
    print("1. 教育语义监控 (专业版，推荐)")
    print("2. 基础文本分析")
    print("3. Web界面模式")
    
    try:
        choice = input("\\n请选择模式 (1/2/3，默认1): ").strip()
        
        if choice == '2':
            start_interactive_mode()
        elif choice == '3':
            start_web_mode()
        else:
            start_educational_monitor()
            
    except KeyboardInterrupt:
        print("\\n\\n👋 用户取消，退出程序")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

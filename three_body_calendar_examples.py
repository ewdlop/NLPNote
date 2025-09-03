#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三體曆法系統使用示例 (Three-Body Calendar System Usage Examples)

展示三體曆法系統的各種使用場景和功能。

Demonstrates various usage scenarios and features of the Three-Body Calendar System.
"""

import datetime
from ThreeBodyCalendar import ThreeBodyCalendar, ThreeBodyDateParser
from three_body_calendar_integration import TemporalExpressionAnalyzer


def example_basic_usage():
    """基本使用示例 - Basic usage example"""
    print("=== 基本使用示例 Basic Usage Example ===\n")
    
    calendar = ThreeBodyCalendar()
    
    # Current time analysis
    now = datetime.datetime.now()
    print("1. 當前時間分析 Current Time Analysis:")
    print(f"Gregorian Date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Chinese format
    print("中文格式 Chinese Format:")
    print(calendar.format_three_body_date(now, 'zh'))
    print()
    
    # English format
    print("English Format:")
    print(calendar.format_three_body_date(now, 'en'))
    print()
    
    # Detailed astronomical information
    state = calendar.get_three_body_state(now)
    print("詳細天文信息 Detailed Astronomical Information:")
    print(f"儒略日 Julian Day: {state.julian_day:.2f}")
    print(f"太陽經度 Sun Longitude: {state.sun.longitude:.2f}°")
    print(f"月球經度 Moon Longitude: {state.moon.longitude:.2f}°")
    print(f"地球偏移 Earth Offset: {state.earth.longitude:.2f}°")
    print()


def example_historical_dates():
    """歷史日期示例 - Historical dates example"""
    print("=== 歷史重要日期 Historical Important Dates ===\n")
    
    calendar = ThreeBodyCalendar()
    
    # Important historical dates
    historical_dates = [
        (datetime.datetime(1969, 7, 20), "Apollo 11 Moon Landing 阿波羅11號登月"),
        (datetime.datetime(2000, 1, 1), "Millennium 千禧年"),
        (datetime.datetime(2024, 4, 8), "Total Solar Eclipse 日全食"),
        (datetime.datetime(2025, 1, 1), "New Year 2025 新年"),
    ]
    
    for dt, description in historical_dates:
        print(f"{description}:")
        print(f"Date: {dt.strftime('%Y-%m-%d')}")
        
        # Three-body calendar analysis
        phase_zh, phase_en, phase_ratio = calendar.lunar_phase(dt)
        term_zh, term_en = calendar.solar_term(dt)
        
        print(f"月相 Lunar Phase: {phase_zh} / {phase_en} ({phase_ratio:.1%})")
        print(f"節氣 Solar Term: {term_zh} / {term_en}")
        print()


def example_cultural_dates():
    """文化節日示例 - Cultural festivals example"""
    print("=== 文化節日分析 Cultural Festival Analysis ===\n")
    
    parser = ThreeBodyDateParser()
    
    cultural_expressions = [
        "2024年春節",
        "2024年清明節", 
        "2024年端午節",
        "2024年中秋節",
        "2025年元宵節"
    ]
    
    for expr in cultural_expressions:
        print(f"分析 Analyzing: {expr}")
        result = parser.analyze_date_expression(expr)
        
        if result['parsed_date']:
            print(f"  解析日期 Parsed Date: {result['parsed_date'].strftime('%Y-%m-%d')}")
            print(f"  月相 Lunar Phase: {result['lunar_phase']['zh']} / {result['lunar_phase']['en']}")
            print(f"  節氣 Solar Term: {result['solar_term']['zh']} / {result['solar_term']['en']}")
        else:
            print("  解析失败 Parsing failed")
        print()


def example_seasonal_analysis():
    """季節變化分析 - Seasonal change analysis"""
    print("=== 2024年季節變化 2024 Seasonal Changes ===\n")
    
    calendar = ThreeBodyCalendar()
    
    # Four seasons and key solar terms
    seasonal_points = [
        (datetime.datetime(2024, 2, 4), "立春 Beginning of Spring"),
        (datetime.datetime(2024, 3, 20), "春分 Spring Equinox"),
        (datetime.datetime(2024, 5, 5), "立夏 Beginning of Summer"),
        (datetime.datetime(2024, 6, 21), "夏至 Summer Solstice"),
        (datetime.datetime(2024, 8, 7), "立秋 Beginning of Autumn"),
        (datetime.datetime(2024, 9, 22), "秋分 Autumn Equinox"),
        (datetime.datetime(2024, 11, 7), "立冬 Beginning of Winter"),
        (datetime.datetime(2024, 12, 21), "冬至 Winter Solstice"),
    ]
    
    print("季節轉換點 Seasonal Transition Points:")
    print("-" * 60)
    
    for dt, description in seasonal_points:
        state = calendar.get_three_body_state(dt)
        phase_zh, phase_en, phase_ratio = calendar.lunar_phase(dt)
        
        print(f"{description}")
        print(f"  日期 Date: {dt.strftime('%Y-%m-%d')}")
        print(f"  儒略日 Julian Day: {state.julian_day:.1f}")
        print(f"  太陽經度 Sun Longitude: {state.sun.longitude:.1f}°")
        print(f"  月相 Lunar Phase: {phase_zh} ({phase_ratio:.1%})")
        print()


def example_nlp_integration():
    """NLP整合示例 - NLP integration example"""
    print("=== NLP整合分析 NLP Integration Analysis ===\n")
    
    analyzer = TemporalExpressionAnalyzer()
    
    # Various temporal expressions
    test_expressions = [
        "2024年冬至是一年中白天最短的一天",
        "明年春分時，晝夜平分",
        "中秋節的月亮最圓最亮", 
        "農曆新年即將到來",
        "The winter solstice marks the beginning of winter",
        "Next full moon will be spectacular"
    ]
    
    for i, expr in enumerate(test_expressions, 1):
        print(f"{i}. 分析表達式 Analyzing Expression:")
        print(f"   Input: '{expr}'")
        
        result = analyzer.analyze_temporal_expression(expr, {
            'situation': 'educational',
            'culture': 'mixed',
            'formality': 'formal'
        })
        
        # Extract key information
        three_body = result['three_body_analysis']
        insights = result['integrated_insights']
        
        if three_body.get('parsed_date'):
            print(f"   解析結果 Parsed: {three_body['parsed_date']}")
            if three_body.get('lunar_phase'):
                print(f"   月相 Lunar Phase: {three_body['lunar_phase']['zh']}")
            if three_body.get('solar_term'):
                print(f"   節氣 Solar Term: {three_body['solar_term']['zh']}")
        else:
            print("   無法解析具體日期 No specific date parsed")
        
        print(f"   天文重要性 Astronomical Importance: {insights['astronomical_importance']}")
        if insights['recommendations']:
            print(f"   建議 Recommendations: {', '.join(insights['recommendations'])}")
        print()


def example_comparative_analysis():
    """比較分析示例 - Comparative analysis example"""
    print("=== 比較分析示例 Comparative Analysis Example ===\n")
    
    analyzer = TemporalExpressionAnalyzer()
    
    # Compare different year's same festivals
    expressions_2024 = [
        "2024年春節",
        "2024年清明節",
        "2024年中秋節", 
        "2024年冬至"
    ]
    
    expressions_2025 = [
        "2025年春節",
        "2025年清明節", 
        "2025年中秋節",
        "2025年冬至"
    ]
    
    print("對比分析：2024年 vs 2025年節日")
    print("Comparative Analysis: 2024 vs 2025 Festivals")
    print("-" * 50)
    
    result_2024 = analyzer.comparative_temporal_analysis(expressions_2024)
    result_2025 = analyzer.comparative_temporal_analysis(expressions_2025)
    
    summary_2024 = result_2024['comparative_analysis']['summary']
    summary_2025 = result_2025['comparative_analysis']['summary']
    
    print("2024年 Year 2024:")
    print(f"  成功解析 Successfully parsed: {summary_2024['successfully_parsed']}/{summary_2024['total_expressions']}")
    print(f"  主要主題 Dominant themes: {summary_2024['dominant_themes']}")
    
    print("\n2025年 Year 2025:")
    print(f"  成功解析 Successfully parsed: {summary_2025['successfully_parsed']}/{summary_2025['total_expressions']}")
    print(f"  主要主題 Dominant themes: {summary_2025['dominant_themes']}")
    
    print("\n天文事件對比 Astronomical Events Comparison:")
    events_2024 = result_2024['comparative_analysis']['astronomical_events']
    events_2025 = result_2025['comparative_analysis']['astronomical_events']
    
    for event in events_2024[:2]:  # Show first 2 events
        print(f"  2024: {event['text']} - {event['phase']['zh']}")
    
    for event in events_2025[:2]:  # Show first 2 events
        print(f"  2025: {event['text']} - {event['phase']['zh']}")


def main():
    """主程序 - Main program"""
    print("=" * 70)
    print("     三體曆法系統完整示例 Three-Body Calendar System Complete Examples")
    print("=" * 70)
    print()
    
    try:
        example_basic_usage()
        print("\n" + "=" * 70 + "\n")
        
        example_historical_dates()
        print("\n" + "=" * 70 + "\n")
        
        example_cultural_dates()
        print("\n" + "=" * 70 + "\n")
        
        example_seasonal_analysis()
        print("\n" + "=" * 70 + "\n")
        
        example_nlp_integration()
        print("\n" + "=" * 70 + "\n")
        
        example_comparative_analysis()
        
        print("\n" + "=" * 70)
        print("示例完成！Examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"錯誤 Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
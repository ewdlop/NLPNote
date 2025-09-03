#!/usr/bin/env python3
"""
日月陽曆(3體)範例 - Solar Lunar Calendar Three-Body System Examples
"""

from SolarLunarCalendar3Body import SolarLunarCalendar3Body, CalendarType
import datetime


def example_basic_usage():
    """基本使用範例 (Basic Usage Example)"""
    print("="*60)
    print("基本使用範例 / Basic Usage Example")
    print("="*60)
    
    # 創建系統實例
    system = SolarLunarCalendar3Body()
    
    # 使用今日日期
    today = datetime.date.today()
    print(f"日期 (Date): {today}")
    
    # 生成英文報告
    report_en = system.generate_multilingual_report(today, "en")
    print("\n英文報告 (English Report):")
    print(report_en)
    
    # 生成中文報告
    report_zh = system.generate_multilingual_report(today, "zh")
    print("\n中文報告 (Chinese Report):")
    print(report_zh)


def example_lunar_phases():
    """月相範例 (Lunar Phase Examples)"""
    print("="*60)
    print("月相範例 / Lunar Phase Examples")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    # 計算未來30天的月相
    today = datetime.date.today()
    
    print("未來30天月相變化 (Lunar Phase Changes for Next 30 Days):")
    print("-" * 50)
    
    for i in range(0, 31, 3):  # 每3天顯示一次
        date = today + datetime.timedelta(days=i)
        lunar_phase = system.lunar_phase_calculation(date)
        
        phase_name_zh = {
            "new_moon": "新月",
            "first_quarter": "上弦月", 
            "full_moon": "滿月",
            "last_quarter": "下弦月"
        }.get(lunar_phase["phase_name"], lunar_phase["phase_name"])
        
        print(f"{date}: {phase_name_zh} ({lunar_phase['phase_name']}) - "
              f"照明: {lunar_phase['illumination_fraction']:.1%}")


def example_calendar_conversion():
    """曆法轉換範例 (Calendar Conversion Examples)"""
    print("="*60)
    print("曆法轉換範例 / Calendar Conversion Examples")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    # 測試幾個特殊日期
    test_dates = [
        datetime.date(2024, 1, 1),   # 新年
        datetime.date(2024, 6, 21),  # 夏至附近
        datetime.date(2024, 12, 21), # 冬至附近
        datetime.date.today()        # 今日
    ]
    
    for date in test_dates:
        print(f"\n日期 (Date): {date}")
        print("-" * 30)
        
        conversion = system.calendar_conversion_table(date)
        
        # 格里高利曆
        gregorian = conversion["gregorian"]
        print(f"格里高利曆: {gregorian['year']}-{gregorian['month']:02d}-{gregorian['day']:02d} "
              f"({gregorian['weekday']})")
        
        # 中國農曆
        chinese = conversion["chinese_lunar"]
        print(f"農曆: {chinese['lunar_year']}年{chinese['lunar_month']}月{chinese['lunar_day']}日 "
              f"({chinese['year_stem_branch']}年)")
        
        # 伊斯蘭曆
        islamic = conversion["islamic"]
        print(f"伊斯蘭曆: {islamic['year']}年{islamic['month']}月{islamic['day']}日")
        
        # 月相
        lunar_phase = conversion["lunar_phase"]
        phase_name_zh = {
            "new_moon": "新月",
            "first_quarter": "上弦月",
            "full_moon": "滿月", 
            "last_quarter": "下弦月"
        }.get(lunar_phase["phase_name"], lunar_phase["phase_name"])
        print(f"月相: {phase_name_zh} (照明: {lunar_phase['illumination_fraction']:.1%})")


def example_three_body_mechanics():
    """三體力學範例 (Three-Body Mechanics Examples)"""
    print("="*60)
    print("三體力學範例 / Three-Body Mechanics Examples")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    # 計算幾個不同時間點的三體位置
    dates = [
        datetime.date(2024, 3, 20),  # 春分
        datetime.date(2024, 6, 21),  # 夏至
        datetime.date(2024, 9, 22),  # 秋分
        datetime.date(2024, 12, 21)  # 冬至
    ]
    
    for date in dates:
        print(f"\n日期 (Date): {date}")
        print("-" * 30)
        
        days_since_epoch = system.solar_calendar_days_since_epoch(date)
        positions = system.three_body_position(days_since_epoch)
        
        print(f"太陽位置 (Sun): {positions.sun_pos}")
        print(f"地球位置 (Earth): {positions.earth_pos}")
        print(f"月球位置 (Moon): {positions.moon_pos}")
        
        # 計算距離
        if hasattr(positions.earth_pos, 'norm'):
            earth_sun_distance = positions.earth_pos.norm()
            moon_earth_distance = (positions.moon_pos - positions.earth_pos).norm()
        else:
            # For numpy arrays
            import math
            earth_sun_distance = math.sqrt(sum(x**2 for x in positions.earth_pos))
            moon_earth_diff = positions.moon_pos - positions.earth_pos
            moon_earth_distance = math.sqrt(sum(x**2 for x in moon_earth_diff))
        
        print(f"地日距離 (Earth-Sun Distance): {earth_sun_distance:.0f} km")
        print(f"月地距離 (Moon-Earth Distance): {moon_earth_distance:.0f} km")
        
        # 計算引力
        forces = system.three_body_forces(positions)
        print(f"地球受力 (Earth Force): {forces['earth']}")


def example_astronomical_events():
    """天文事件範例 (Astronomical Events Examples)"""
    print("="*60)
    print("天文事件範例 / Astronomical Events Examples")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    # 預測未來一年的日食月食
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=365)
    
    print(f"日食月食預測 ({today} 到 {end_date}):")
    print("-" * 50)
    
    eclipses = system.eclipse_prediction(today, end_date)
    
    for eclipse in eclipses[:10]:  # 顯示前10個
        eclipse_type_zh = {
            "solar_eclipse": "日食",
            "lunar_eclipse": "月食"
        }.get(eclipse.name, eclipse.name)
        
        print(f"{eclipse.date}: {eclipse_type_zh} ({eclipse.name})")
        print(f"  說明: {eclipse.description['zh']}")
        print(f"  Description: {eclipse.description['en']}")
        
    # 計算二十四節氣
    print(f"\n{today.year}年二十四節氣:")
    print("-" * 50)
    
    solar_terms = system.solar_terms_calculation(today.year)
    
    for term in solar_terms:
        print(f"{term.date}: {term.description['zh']} ({term.description['en']})")


def example_multilingual_terminology():
    """多語言術語範例 (Multilingual Terminology Examples)"""
    print("="*60)
    print("多語言術語範例 / Multilingual Terminology Examples")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    # 主要術語
    main_terms = [
        "solar_calendar",
        "lunar_calendar", 
        "three_body_system",
        "new_moon",
        "full_moon",
        "solar_eclipse",
        "lunar_eclipse"
    ]
    
    # 支援的語言
    languages = [
        ("en", "English"),
        ("zh", "中文"),
        ("ja", "日本語"),
        ("ko", "한국어"),
        ("fr", "Français"),
        ("de", "Deutsch"),
        ("es", "Español"),
        ("ru", "Русский"),
        ("ar", "العربية")
    ]
    
    for term in main_terms:
        print(f"\n術語 (Term): {term}")
        print("-" * 30)
        
        for lang_code, lang_name in languages:
            translation = system.get_term(term, lang_code)
            print(f"{lang_name:10} ({lang_code}): {translation}")


def example_interactive_demo():
    """互動演示 (Interactive Demo)"""
    print("="*60)
    print("互動演示 / Interactive Demo")
    print("="*60)
    
    system = SolarLunarCalendar3Body()
    
    while True:
        print("\n選擇功能 (Choose Function):")
        print("1. 查看特定日期曆法信息 (View calendar info for specific date)")
        print("2. 月相計算 (Lunar phase calculation)")
        print("3. 三體位置 (Three-body positions)")
        print("4. 多語言術語 (Multilingual terms)")
        print("0. 退出 (Exit)")
        
        try:
            choice = input("\n請輸入選擇 (Enter choice): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                date_str = input("輸入日期 (YYYY-MM-DD): ").strip()
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    report = system.generate_multilingual_report(date, "zh")
                    print(report)
                except ValueError:
                    print("日期格式錯誤！")
            elif choice == "2":
                date_str = input("輸入日期 (YYYY-MM-DD): ").strip()
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    lunar_phase = system.lunar_phase_calculation(date)
                    phase_name_zh = {
                        "new_moon": "新月",
                        "first_quarter": "上弦月",
                        "full_moon": "滿月",
                        "last_quarter": "下弦月"
                    }.get(lunar_phase["phase_name"], lunar_phase["phase_name"])
                    
                    print(f"月相: {phase_name_zh}")
                    print(f"照明比例: {lunar_phase['illumination_fraction']:.2%}")
                    print(f"月相角度: {lunar_phase['phase_angle']:.1f}°")
                except ValueError:
                    print("日期格式錯誤！")
            elif choice == "3":
                date_str = input("輸入日期 (YYYY-MM-DD): ").strip()
                try:
                    date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                    days_since_epoch = system.solar_calendar_days_since_epoch(date)
                    positions = system.three_body_position(days_since_epoch)
                    
                    print(f"太陽位置: {positions.sun_pos}")
                    print(f"地球位置: {positions.earth_pos}")
                    print(f"月球位置: {positions.moon_pos}")
                except ValueError:
                    print("日期格式錯誤！")
            elif choice == "4":
                term = input("輸入術語 (例如: solar_calendar): ").strip()
                lang = input("輸入語言代碼 (例如: zh, en, ja): ").strip()
                translation = system.get_term(term, lang)
                print(f"{term} ({lang}): {translation}")
            else:
                print("無效選擇！")
                
        except KeyboardInterrupt:
            print("\n\n再見！")
            break
        except Exception as e:
            print(f"錯誤: {e}")


def main():
    """主函數 (Main Function)"""
    print("日月陽曆(3體)系統範例演示")
    print("Solar Lunar Calendar Three-Body System Examples")
    print("="*80)
    
    # 運行所有範例
    example_basic_usage()
    print("\n" + "="*80 + "\n")
    
    example_lunar_phases()
    print("\n" + "="*80 + "\n")
    
    example_calendar_conversion()
    print("\n" + "="*80 + "\n")
    
    example_three_body_mechanics()
    print("\n" + "="*80 + "\n")
    
    example_astronomical_events()
    print("\n" + "="*80 + "\n")
    
    example_multilingual_terminology()
    print("\n" + "="*80 + "\n")
    
    # 可選的互動模式
    try:
        response = input("是否進入互動模式？(y/N): ").strip().lower()
        if response in ['y', 'yes', '是']:
            example_interactive_demo()
    except KeyboardInterrupt:
        print("\n程序結束")


if __name__ == "__main__":
    main()
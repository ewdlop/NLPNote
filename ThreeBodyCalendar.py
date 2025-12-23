#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三體曆法系統 (Three-Body Calendar System)
地日月曆 - Earth-Sun-Moon Calendar

A calendar system that accounts for the gravitational interactions between 
Earth, Sun, and Moon with natural language processing capabilities.

Author: Three-Body Calendar Development Team
Date: 2024-12-22
"""

import math
import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re


@dataclass
class CelestialPosition:
    """天體位置 - Celestial body position"""
    longitude: float  # 經度 (degrees)
    latitude: float   # 緯度 (degrees) 
    distance: float   # 距離 (astronomical units)
    
    def __str__(self):
        return f"位置(經度={self.longitude:.2f}°, 緯度={self.latitude:.2f}°, 距離={self.distance:.4f}AU)"


@dataclass
class ThreeBodyState:
    """三體狀態 - Three-body system state"""
    earth: CelestialPosition
    sun: CelestialPosition  
    moon: CelestialPosition
    julian_day: float
    gregorian_date: datetime.datetime
    
    def __str__(self):
        return (f"三體狀態 {self.gregorian_date.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  地球: {self.earth}\n"
                f"  太陽: {self.sun}\n" 
                f"  月球: {self.moon}")


class AstronomicalCalculator:
    """天文計算器 - Astronomical calculator for celestial mechanics"""
    
    @staticmethod
    def julian_day(dt: datetime.datetime) -> float:
        """計算儒略日 - Calculate Julian Day Number"""
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        
        jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
        
        # Add fractional day
        fraction = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
        return jdn + fraction - 0.5
    
    @staticmethod
    def sun_position(jd: float) -> CelestialPosition:
        """計算太陽位置 - Calculate Sun position (simplified)"""
        # Simplified solar position calculation
        n = jd - 2451545.0  # Days since J2000.0
        L = (280.460 + 0.9856474 * n) % 360  # Mean longitude
        g = math.radians((357.528 + 0.9856003 * n) % 360)  # Mean anomaly
        
        # Equation of center (simplified)
        C = 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)
        longitude = (L + C) % 360
        
        # Distance (simplified)
        distance = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)
        
        return CelestialPosition(longitude=longitude, latitude=0.0, distance=distance)
    
    @staticmethod  
    def moon_position(jd: float) -> CelestialPosition:
        """計算月球位置 - Calculate Moon position (simplified)"""
        # Simplified lunar position calculation
        n = jd - 2451545.0
        
        # Mean elements
        L = (218.316 + 13.176396 * n) % 360  # Mean longitude
        M = math.radians((134.963 + 13.064993 * n) % 360)  # Mean anomaly
        F = math.radians((93.272 + 13.229350 * n) % 360)   # Mean distance from ascending node
        
        # Simplified perturbations
        longitude = L + 6.289 * math.sin(M) + 1.274 * math.sin(2 * F - M) + 0.658 * math.sin(2 * F)
        longitude = longitude % 360
        
        latitude = 5.128 * math.sin(F) + 0.280 * math.sin(M + F) + 0.277 * math.sin(M - F)
        
        # Distance in Earth radii (convert to AU)
        distance = (385000.56 + 20905.36 * math.cos(M) + 3699.11 * math.cos(2 * F - M)) / 149597870.7
        
        return CelestialPosition(longitude=longitude, latitude=latitude, distance=distance)
    
    @staticmethod
    def earth_position(jd: float) -> CelestialPosition:
        """計算地球位置 - Calculate Earth position (relative to solar system barycenter)"""
        # For simplicity, assume Earth is at origin with small perturbations due to Moon
        moon_pos = AstronomicalCalculator.moon_position(jd)
        
        # Earth-Moon barycenter offset (simplified)
        offset_factor = 0.012  # Approximate ratio
        longitude = (moon_pos.longitude + 180) % 360  # Opposite to Moon's apparent position
        latitude = -moon_pos.latitude * offset_factor
        distance = 1.0 + moon_pos.distance * offset_factor * 0.001
        
        return CelestialPosition(longitude=longitude, latitude=latitude, distance=distance)


class ThreeBodyCalendar:
    """三體曆法 - Three-Body Calendar System"""
    
    def __init__(self):
        self.calculator = AstronomicalCalculator()
        self._lunar_month_days = 29.53059  # 朔望月長度 - Synodic month length
        self._solar_year_days = 365.24219  # 回歸年長度 - Tropical year length
        
    def get_three_body_state(self, dt: datetime.datetime) -> ThreeBodyState:
        """獲取三體狀態 - Get three-body system state for given date"""
        jd = self.calculator.julian_day(dt)
        
        earth_pos = self.calculator.earth_position(jd)
        sun_pos = self.calculator.sun_position(jd)
        moon_pos = self.calculator.moon_position(jd)
        
        return ThreeBodyState(
            earth=earth_pos,
            sun=sun_pos,
            moon=moon_pos,
            julian_day=jd,
            gregorian_date=dt
        )
    
    def lunar_phase(self, dt: datetime.datetime) -> Tuple[str, str, float]:
        """計算月相 - Calculate lunar phase"""
        state = self.get_three_body_state(dt)
        
        # Calculate phase angle
        sun_moon_angle = abs(state.sun.longitude - state.moon.longitude) % 360
        if sun_moon_angle > 180:
            sun_moon_angle = 360 - sun_moon_angle
            
        phase_ratio = (1 - math.cos(math.radians(sun_moon_angle))) / 2
        
        # Determine phase name
        if phase_ratio < 0.1:
            phase_zh, phase_en = "新月", "New Moon"
        elif phase_ratio < 0.4:
            phase_zh, phase_en = "眉月", "Waxing Crescent"
        elif 0.4 <= phase_ratio < 0.6:
            phase_zh, phase_en = "上弦月", "First Quarter"
        elif 0.6 <= phase_ratio < 0.9:
            phase_zh, phase_en = "盈凸月", "Waxing Gibbous"
        elif phase_ratio >= 0.9:
            phase_zh, phase_en = "滿月", "Full Moon"
        else:
            phase_zh, phase_en = "殘月", "Waning"
            
        return phase_zh, phase_en, phase_ratio
    
    def solar_term(self, dt: datetime.datetime) -> Tuple[str, str]:
        """計算節氣 - Calculate solar term (Chinese 24 solar terms)"""
        state = self.get_three_body_state(dt)
        sun_longitude = state.sun.longitude
        
        # Solar terms based on sun's longitude
        solar_terms_zh = [
            "春分", "清明", "穀雨", "立夏", "小滿", "芒種",
            "夏至", "小暑", "大暑", "立秋", "處暑", "白露", 
            "秋分", "寒露", "霜降", "立冬", "小雪", "大雪",
            "冬至", "小寒", "大寒", "立春", "雨水", "驚蟄"
        ]
        
        solar_terms_en = [
            "Spring Equinox", "Clear and Bright", "Grain Rain", "Beginning of Summer",
            "Grain Buds", "Grain in Ear", "Summer Solstice", "Slight Heat",
            "Great Heat", "Beginning of Autumn", "Stopping the Heat", "White Dews",
            "Autumn Equinox", "Cold Dews", "Frost's Descent", "Beginning of Winter", 
            "Slight Snow", "Great Snow", "Winter Solstice", "Slight Cold",
            "Great Cold", "Beginning of Spring", "Rain Water", "Awakening of Insects"
        ]
        
        # Each solar term spans 15 degrees
        term_index = int(sun_longitude // 15) % 24
        return solar_terms_zh[term_index], solar_terms_en[term_index]
    
    def format_three_body_date(self, dt: datetime.datetime, lang: str = "zh") -> str:
        """格式化三體日期 - Format date in three-body calendar system"""
        state = self.get_three_body_state(dt)
        phase_zh, phase_en, phase_ratio = self.lunar_phase(dt)
        term_zh, term_en = self.solar_term(dt)
        
        if lang == "zh":
            return (f"三體曆 {dt.year}年{dt.month}月{dt.day}日\n"
                   f"節氣: {term_zh}\n"
                   f"月相: {phase_zh} ({phase_ratio:.1%})\n"
                   f"儒略日: {state.julian_day:.1f}")
        else:
            return (f"Three-Body Calendar {dt.strftime('%Y-%m-%d')}\n"
                   f"Solar Term: {term_en}\n" 
                   f"Lunar Phase: {phase_en} ({phase_ratio:.1%})\n"
                   f"Julian Day: {state.julian_day:.1f}")


class ThreeBodyDateParser:
    """三體日期解析器 - Natural language date parser for three-body calendar"""
    
    def __init__(self):
        self.calendar = ThreeBodyCalendar()
        
        # Chinese date patterns
        self.zh_patterns = {
            'year': r'(\d{4})年',
            'month': r'(\d{1,2})月', 
            'day': r'(\d{1,2})日',
            'lunar_phase': r'(新月|眉月|上弦月|盈凸月|滿月|殘月)',
            'solar_term': r'(春分|清明|穀雨|立夏|小滿|芒種|夏至|小暑|大暑|立秋|處暑|白露|秋分|寒露|霜降|立冬|小雪|大雪|冬至|小寒|大寒|立春|雨水|驚蟄)',
            'year_only': r'(\d{4})年',
            'solar_term_with_year': r'(\d{4})年(\S*)(春分|清明|穀雨|立夏|小滿|芒種|夏至|小暑|大暑|立秋|處暑|白露|秋分|寒露|霜降|立冬|小雪|大雪|冬至|小寒|大寒|立春|雨水|驚蟄)',
            'relative_lunar': r'(下個|下一個|下次|今天|明天|後天)(滿月|新月|眉月|上弦月|盈凸月|殘月)',
            'festival': r'(中秋節|春節|元宵節|清明節|端午節|七夕|重陽節)'
        }
        
        # English date patterns  
        self.en_patterns = {
            'date': r'(\d{4})-(\d{1,2})-(\d{1,2})',
            'lunar_phase': r'(New Moon|Waxing Crescent|First Quarter|Waxing Gibbous|Full Moon|Waning)',
            'solar_term': r'(Spring Equinox|Clear and Bright|Grain Rain|Beginning of Summer|Grain Buds|Grain in Ear|Summer Solstice|Slight Heat|Great Heat|Beginning of Autumn|Stopping the Heat|White Dews|Autumn Equinox|Cold Dews|Frost\'s Descent|Beginning of Winter|Slight Snow|Great Snow|Winter Solstice|Slight Cold|Great Cold|Beginning of Spring|Rain Water|Awakening of Insects)'
        }
        
    def parse_chinese_date(self, text: str) -> Optional[datetime.datetime]:
        """解析中文日期 - Parse Chinese date expression"""
        # Try full date first
        year_match = re.search(self.zh_patterns['year'], text)
        month_match = re.search(self.zh_patterns['month'], text) 
        day_match = re.search(self.zh_patterns['day'], text)
        
        if year_match and month_match and day_match:
            try:
                year = int(year_match.group(1))
                month = int(month_match.group(1))
                day = int(day_match.group(1))
                return datetime.datetime(year, month, day)
            except ValueError:
                return None
        
        # Try solar term with year
        solar_term_match = re.search(self.zh_patterns['solar_term_with_year'], text)
        if solar_term_match:
            try:
                year = int(solar_term_match.group(1))
                solar_term = solar_term_match.group(3)
                # Estimate date based on solar term
                estimated_date = self._estimate_solar_term_date(year, solar_term)
                return estimated_date
            except ValueError:
                return None
        
        # Try solar term for current year
        solar_term_match = re.search(self.zh_patterns['solar_term'], text)
        if solar_term_match:
            solar_term = solar_term_match.group(1)
            current_year = datetime.datetime.now().year
            estimated_date = self._estimate_solar_term_date(current_year, solar_term)
            return estimated_date
        
        # Try festivals
        festival_match = re.search(self.zh_patterns['festival'], text)
        if festival_match:
            festival = festival_match.group(1)
            current_year = datetime.datetime.now().year
            estimated_date = self._estimate_festival_date(current_year, festival)
            return estimated_date
            
        return None
        
    def parse_english_date(self, text: str) -> Optional[datetime.datetime]:
        """Parse English date expression"""
        date_match = re.search(self.en_patterns['date'], text)
        
        if date_match:
            try:
                year = int(date_match.group(1))
                month = int(date_match.group(2))
                day = int(date_match.group(3))
                return datetime.datetime(year, month, day)
            except ValueError:
                return None
        return None
        
    def parse_date(self, text: str) -> Optional[datetime.datetime]:
        """解析日期文本 - Parse date from text in multiple languages"""
        # Try Chinese first
        result = self.parse_chinese_date(text)
        if result:
            return result
            
        # Try English  
        result = self.parse_english_date(text)
        if result:
            return result
            
        return None
    
    def _estimate_solar_term_date(self, year: int, solar_term: str) -> Optional[datetime.datetime]:
        """估算節氣日期 - Estimate solar term date"""
        # Mapping of solar terms to approximate dates
        solar_term_dates = {
            '立春': (2, 4), '雨水': (2, 19), '驚蟄': (3, 6), '春分': (3, 21),
            '清明': (4, 5), '穀雨': (4, 20), '立夏': (5, 6), '小滿': (5, 21),
            '芒種': (6, 6), '夏至': (6, 21), '小暑': (7, 7), '大暑': (7, 23),
            '立秋': (8, 8), '處暑': (8, 23), '白露': (9, 8), '秋分': (9, 23),
            '寒露': (10, 8), '霜降': (10, 23), '立冬': (11, 8), '小雪': (11, 22),
            '大雪': (12, 7), '冬至': (12, 22), '小寒': (1, 6), '大寒': (1, 20)
        }
        
        if solar_term in solar_term_dates:
            month, day = solar_term_dates[solar_term]
            try:
                return datetime.datetime(year, month, day)
            except ValueError:
                return None
        return None
    
    def _estimate_festival_date(self, year: int, festival: str) -> Optional[datetime.datetime]:
        """估算節日日期 - Estimate festival date"""
        # Simplified festival date estimation
        festival_dates = {
            '春節': (2, 1),    # Approximate
            '元宵節': (2, 15),  # Approximate  
            '清明節': (4, 5),   # Around Qingming solar term
            '端午節': (6, 15),  # Approximate
            '中秋節': (9, 15),  # Mid-autumn, approximate
            '七夕': (8, 15),    # Approximate
            '重陽節': (10, 15)  # Approximate
        }
        
        if festival in festival_dates:
            month, day = festival_dates[festival]
            try:
                return datetime.datetime(year, month, day)
            except ValueError:
                return None
        return None
        
    def analyze_date_expression(self, text: str) -> Dict[str, Union[str, datetime.datetime, None]]:
        """分析日期表達式 - Analyze date expression with three-body calendar context"""
        parsed_date = self.parse_date(text)
        
        result = {
            'input_text': text,
            'parsed_date': parsed_date,
            'three_body_format_zh': None,
            'three_body_format_en': None,
            'lunar_phase': None,
            'solar_term': None
        }
        
        if parsed_date:
            result['three_body_format_zh'] = self.calendar.format_three_body_date(parsed_date, 'zh')
            result['three_body_format_en'] = self.calendar.format_three_body_date(parsed_date, 'en')
            
            phase_zh, phase_en, _ = self.calendar.lunar_phase(parsed_date)
            result['lunar_phase'] = {'zh': phase_zh, 'en': phase_en}
            
            term_zh, term_en = self.calendar.solar_term(parsed_date)
            result['solar_term'] = {'zh': term_zh, 'en': term_en}
            
        return result


def demo_three_body_calendar():
    """演示三體曆法系統 - Demo the three-body calendar system"""
    print("=== 三體曆法系統演示 Three-Body Calendar System Demo ===\n")
    
    calendar = ThreeBodyCalendar()
    parser = ThreeBodyDateParser()
    
    # Current date
    now = datetime.datetime.now()
    print("1. 當前時間 Current Time:")
    print(calendar.format_three_body_date(now, 'zh'))
    print()
    print(calendar.format_three_body_date(now, 'en'))
    print("\n" + "="*50 + "\n")
    
    # Historical dates
    historical_dates = [
        (datetime.datetime(2024, 3, 20), "春分 Spring Equinox"),
        (datetime.datetime(2024, 6, 21), "夏至 Summer Solstice"), 
        (datetime.datetime(2024, 9, 22), "秋分 Autumn Equinox"),
        (datetime.datetime(2024, 12, 21), "冬至 Winter Solstice")
    ]
    
    print("2. 重要節氣 Important Solar Terms:")
    for dt, description in historical_dates:
        print(f"\n{description}:")
        print(calendar.format_three_body_date(dt, 'zh'))
        print()
        
    print("="*50 + "\n")
    
    # NLP parsing demo
    print("3. 自然語言解析演示 Natural Language Parsing Demo:")
    test_expressions = [
        "2024年12月22日",
        "2025-01-01", 
        "今天是冬至",
        "Next full moon"
    ]
    
    for expr in test_expressions:
        print(f"\n輸入 Input: '{expr}'")
        analysis = parser.analyze_date_expression(expr)
        if analysis['parsed_date']:
            print(f"解析結果 Parsed: {analysis['parsed_date']}")
            print(f"月相 Lunar Phase: {analysis['lunar_phase']}")
            print(f"節氣 Solar Term: {analysis['solar_term']}")
        else:
            print("無法解析 Cannot parse")


if __name__ == "__main__":
    demo_three_body_calendar()
"""
日月陽曆(3體) - Solar Lunar Calendar Three-Body System
太陽月亮陽曆三體系統

This module implements a comprehensive system for:
1. Solar and Lunar calendar calculations
2. Three-body astronomical computations
3. Multilingual calendar terminology
4. NLP processing for calendar-related text

本模組實現以下功能:
1. 太陽曆和陰曆計算
2. 三體天體力學計算
3. 多語言曆法術語
4. 曆法相關文本的自然語言處理
"""

import math
import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import calendar

# Handle numpy dependency gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple numpy-like functionality
    class SimpleArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = list(data)
            else:
                self.data = [data]
        
        def __add__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a + b for a, b in zip(self.data, other.data)])
            return SimpleArray([a + other for a in self.data])
        
        def __sub__(self, other):
            if isinstance(other, SimpleArray):
                return SimpleArray([a - b for a, b in zip(self.data, other.data)])
            return SimpleArray([a - other for a in self.data])
        
        def __mul__(self, scalar):
            return SimpleArray([a * scalar for a in self.data])
        
        def __rmul__(self, scalar):
            return self.__mul__(scalar)
        
        def __neg__(self):
            return SimpleArray([-a for a in self.data])
        
        def __getitem__(self, index):
            return self.data[index]
        
        def __setitem__(self, index, value):
            self.data[index] = value
        
        def norm(self):
            return math.sqrt(sum(x**2 for x in self.data))
        
        def normalize(self):
            n = self.norm()
            if n == 0:
                return SimpleArray(self.data)
            return SimpleArray([x/n for x in self.data])
        
        def __repr__(self):
            return f"SimpleArray({self.data})"
    
    # Create a simple numpy-like interface
    class np:
        @staticmethod
        def array(data):
            return SimpleArray(data)
        
        @staticmethod  
        def linalg_norm(arr):
            return arr.norm()

class CalendarType(Enum):
    """曆法類型 (Calendar Types)"""
    SOLAR = "solar"  # 陽曆/太陽曆
    LUNAR = "lunar"  # 陰曆/太陰曆  
    LUNISOLAR = "lunisolar"  # 陰陽合曆
    GREGORIAN = "gregorian"  # 格里高利曆
    JULIAN = "julian"  # 儒略曆
    CHINESE = "chinese"  # 中國農曆
    ISLAMIC = "islamic"  # 伊斯蘭曆
    HEBREW = "hebrew"  # 希伯來曆

class CelestialBody(Enum):
    """天體 (Celestial Bodies)"""
    SUN = "sun"      # 太陽
    MOON = "moon"    # 月亮
    EARTH = "earth"  # 地球

@dataclass
class AstronomicalData:
    """天文數據 (Astronomical Data)"""
    solar_year: float = 365.25636  # 回歸年 (tropical year) in days
    lunar_month: float = 29.530589  # 朔望月 (synodic month) in days
    sidereal_month: float = 27.321661  # 恆星月 (sidereal month) in days
    earth_orbit_radius: float = 149597870.7  # 地球軌道半徑 (km)
    moon_orbit_radius: float = 384400  # 月球軌道半徑 (km)
    sun_mass: float = 1.989e30  # 太陽質量 (kg)
    earth_mass: float = 5.972e24  # 地球質量 (kg)
    moon_mass: float = 7.342e22  # 月球質量 (kg)

@dataclass
class CalendarEvent:
    """曆法事件 (Calendar Event)"""
    name: str
    date: datetime.date
    calendar_type: CalendarType
    description: Dict[str, str]  # 多語言描述
    astronomical_significance: Optional[str] = None

@dataclass
class ThreeBodyPosition:
    """三體位置 (Three-Body Position)"""
    sun_pos: Union['SimpleArray', 'np.ndarray']
    earth_pos: Union['SimpleArray', 'np.ndarray'] 
    moon_pos: Union['SimpleArray', 'np.ndarray']
    time: float  # 時間 (days since epoch)

class SolarLunarCalendar3Body:
    """日月陽曆三體系統 (Solar Lunar Calendar Three-Body System)"""
    
    def __init__(self):
        self.astro_data = AstronomicalData()
        self.calendar_terms = self._load_calendar_terminology()
        self.gravitational_constant = 6.67430e-11  # G in m^3 kg^-1 s^-2
        
    def _load_calendar_terminology(self) -> Dict[str, Dict[str, str]]:
        """載入多語言曆法術語 (Load multilingual calendar terminology)"""
        return {
            "solar_calendar": {
                "en": "Solar Calendar",
                "zh": "陽曆",
                "zh-tw": "陽曆", 
                "ja": "太陽暦",
                "ko": "양력",
                "fr": "Calendrier solaire",
                "de": "Sonnenkalender",
                "es": "Calendario solar",
                "ru": "Солнечный календарь",
                "ar": "التقويم الشمسي"
            },
            "lunar_calendar": {
                "en": "Lunar Calendar",
                "zh": "陰曆", 
                "zh-tw": "陰曆",
                "ja": "太陰暦",
                "ko": "음력",
                "fr": "Calendrier lunaire",
                "de": "Mondkalender", 
                "es": "Calendario lunar",
                "ru": "Лунный календарь",
                "ar": "التقويم القمري"
            },
            "three_body_system": {
                "en": "Three-Body System",
                "zh": "三體系統",
                "zh-tw": "三體系統", 
                "ja": "三体系",
                "ko": "삼체계",
                "fr": "Système à trois corps",
                "de": "Dreikörpersystem",
                "es": "Sistema de tres cuerpos",
                "ru": "Система трех тел",
                "ar": "نظام الأجسام الثلاثة"
            },
            "new_moon": {
                "en": "New Moon",
                "zh": "新月",
                "zh-tw": "新月",
                "ja": "新月",
                "ko": "신월",
                "fr": "Nouvelle lune",
                "de": "Neumond",
                "es": "Luna nueva", 
                "ru": "Новолуние",
                "ar": "القمر الجديد"
            },
            "full_moon": {
                "en": "Full Moon",
                "zh": "滿月",
                "zh-tw": "滿月",
                "ja": "満月", 
                "ko": "보름달",
                "fr": "Pleine lune",
                "de": "Vollmond",
                "es": "Luna llena",
                "ru": "Полнолуние", 
                "ar": "القمر المكتمل"
            },
            "solar_eclipse": {
                "en": "Solar Eclipse",
                "zh": "日食",
                "zh-tw": "日食",
                "ja": "日食",
                "ko": "일식",
                "fr": "Éclipse solaire",
                "de": "Sonnenfinsternis",
                "es": "Eclipse solar",
                "ru": "Солнечное затмение",
                "ar": "كسوف الشمس"
            },
            "lunar_eclipse": {
                "en": "Lunar Eclipse",
                "zh": "月食",
                "zh-tw": "月食", 
                "ja": "月食",
                "ko": "월식",
                "fr": "Éclipse lunaire",
                "de": "Mondfinsternis",
                "es": "Eclipse lunar",
                "ru": "Лунное затмение",
                "ar": "خسوف القمر"
            }
        }
    
    def get_term(self, term: str, language: str = "en") -> str:
        """獲取指定語言的曆法術語 (Get calendar term in specified language)"""
        if term in self.calendar_terms and language in self.calendar_terms[term]:
            return self.calendar_terms[term][language]
        return term  # 回退到原術語
    
    def solar_calendar_days_since_epoch(self, date: datetime.date) -> float:
        """計算自紀元以來的太陽曆天數 (Calculate solar calendar days since epoch)"""
        epoch = datetime.date(2000, 1, 1)  # J2000.0 epoch
        delta = date - epoch
        return delta.days + delta.seconds / 86400.0
    
    def lunar_phase_calculation(self, date: datetime.date) -> Dict[str, Union[float, str]]:
        """計算月相 (Calculate lunar phase)"""
        days_since_epoch = self.solar_calendar_days_since_epoch(date)
        
        # 新月週期計算 (New moon cycle calculation)
        lunar_cycle = days_since_epoch / self.astro_data.lunar_month
        phase_in_cycle = lunar_cycle - math.floor(lunar_cycle)
        
        # 月相角度 (Phase angle in degrees)
        phase_angle = phase_in_cycle * 360.0
        
        # 月相名稱 (Phase name)
        if phase_angle < 45 or phase_angle >= 315:
            phase_name = "new_moon"
        elif 45 <= phase_angle < 135:
            phase_name = "first_quarter"
        elif 135 <= phase_angle < 225:
            phase_name = "full_moon"
        else:
            phase_name = "last_quarter"
        
        # 照明比例 (Illumination fraction)
        illumination = 0.5 * (1 - math.cos(math.radians(phase_angle)))
        
        return {
            "phase_angle": phase_angle,
            "phase_name": phase_name,
            "illumination_fraction": illumination,
            "days_since_new_moon": phase_in_cycle * self.astro_data.lunar_month
        }
    
    def chinese_calendar_conversion(self, date: datetime.date) -> Dict[str, Union[int, str]]:
        """中國農曆轉換 (Chinese lunar calendar conversion)"""
        # 簡化的農曆計算 (Simplified lunar calendar calculation)
        days_since_epoch = self.solar_calendar_days_since_epoch(date)
        
        # 農曆年計算 (Lunar year calculation)  
        lunar_year_length = 12 * self.astro_data.lunar_month
        approximate_lunar_year = 2000 + int(days_since_epoch / lunar_year_length)
        
        # 農曆月計算 (Lunar month calculation)
        lunar_month_num = int((days_since_epoch % lunar_year_length) / self.astro_data.lunar_month) + 1
        
        # 農曆日計算 (Lunar day calculation)
        lunar_day = int(days_since_epoch % self.astro_data.lunar_month) + 1
        
        # 天干地支 (Heavenly Stems and Earthly Branches)
        heavenly_stems = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
        earthly_branches = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
        
        year_stem = heavenly_stems[(approximate_lunar_year - 4) % 10]
        year_branch = earthly_branches[(approximate_lunar_year - 4) % 12]
        
        return {
            "lunar_year": approximate_lunar_year,
            "lunar_month": lunar_month_num,
            "lunar_day": lunar_day,
            "year_stem_branch": f"{year_stem}{year_branch}",
            "year_stem": year_stem,
            "year_branch": year_branch
        }
    
    def three_body_position(self, time_days: float) -> ThreeBodyPosition:
        """計算日地月三體位置 (Calculate Sun-Earth-Moon three-body positions)"""
        # 時間轉換為弧度 (Time conversion to radians)
        t = time_days / 365.25  # 年為單位 (in years)
        
        # 地球繞太陽運動 (Earth's orbit around Sun)
        earth_angle = 2 * math.pi * t  # 地球軌道角度
        earth_pos = np.array([
            self.astro_data.earth_orbit_radius * math.cos(earth_angle),
            self.astro_data.earth_orbit_radius * math.sin(earth_angle),
            0
        ])
        
        # 月球繞地球運動 (Moon's orbit around Earth)  
        moon_orbital_period = self.astro_data.lunar_month / 365.25  # 月球軌道週期(年)
        moon_angle = 2 * math.pi * t / moon_orbital_period
        moon_relative_pos = np.array([
            self.astro_data.moon_orbit_radius * math.cos(moon_angle),
            self.astro_data.moon_orbit_radius * math.sin(moon_angle),
            0
        ])
        moon_pos = earth_pos + moon_relative_pos
        
        # 太陽位置 (Sun position at origin)
        sun_pos = np.array([0, 0, 0])
        
        return ThreeBodyPosition(sun_pos, earth_pos, moon_pos, time_days)
    
    def gravitational_force(self, pos1: Union['SimpleArray', 'np.ndarray'], mass1: float, 
                          pos2: Union['SimpleArray', 'np.ndarray'], mass2: float) -> Union['SimpleArray', 'np.ndarray']:
        """計算兩體間引力 (Calculate gravitational force between two bodies)"""
        r_vec = pos2 - pos1
        if NUMPY_AVAILABLE:
            r_mag = np.linalg.norm(r_vec)
        else:
            r_mag = r_vec.norm()
        
        if r_mag == 0:
            return np.array([0, 0, 0]) if NUMPY_AVAILABLE else SimpleArray([0, 0, 0])
            
        if NUMPY_AVAILABLE:
            r_unit = r_vec / r_mag
        else:
            r_unit = r_vec.normalize()
        
        # Convert to meters for calculation
        r_mag_meters = r_mag * 1000  # km to m
        force_magnitude = self.gravitational_constant * mass1 * mass2 / (r_mag_meters ** 2)
        force_vector = force_magnitude * r_unit
        
        return force_vector
    
    def three_body_forces(self, positions: ThreeBodyPosition) -> Dict[str, Union['SimpleArray', 'np.ndarray']]:
        """計算三體系統中的引力 (Calculate forces in three-body system)"""
        masses = {
            'sun': self.astro_data.sun_mass,
            'earth': self.astro_data.earth_mass, 
            'moon': self.astro_data.moon_mass
        }
        
        # 地球受力 (Forces on Earth)
        force_earth_from_sun = self.gravitational_force(
            positions.earth_pos, masses['earth'],
            positions.sun_pos, masses['sun']
        )
        force_earth_from_moon = self.gravitational_force(
            positions.earth_pos, masses['earth'],
            positions.moon_pos, masses['moon']
        )
        total_force_earth = force_earth_from_sun + force_earth_from_moon
        
        # 月球受力 (Forces on Moon)
        force_moon_from_sun = self.gravitational_force(
            positions.moon_pos, masses['moon'],
            positions.sun_pos, masses['sun']
        )
        force_moon_from_earth = self.gravitational_force(
            positions.moon_pos, masses['moon'],
            positions.earth_pos, masses['earth']
        )
        total_force_moon = force_moon_from_sun + force_moon_from_earth
        
        # 太陽受力 (Forces on Sun)
        force_sun_from_earth = -force_earth_from_sun  # 牛頓第三定律
        force_sun_from_moon = -force_moon_from_sun
        total_force_sun = force_sun_from_earth + force_sun_from_moon
        
        return {
            'sun': total_force_sun,
            'earth': total_force_earth,
            'moon': total_force_moon
        }
    
    def eclipse_prediction(self, start_date: datetime.date, 
                          end_date: datetime.date) -> List[CalendarEvent]:
        """預測日食月食 (Predict solar and lunar eclipses)"""
        events = []
        current_date = start_date
        
        while current_date <= end_date:
            # 檢查月相 (Check lunar phase)
            lunar_phase = self.lunar_phase_calculation(current_date)
            
            # 簡化的日食預測 (Simplified solar eclipse prediction)
            if lunar_phase["phase_name"] == "new_moon":
                # 日食可能發生在新月時 (Solar eclipses occur during new moon)
                if abs(lunar_phase["days_since_new_moon"]) < 0.5:  # 在新月附近
                    eclipse_event = CalendarEvent(
                        name="solar_eclipse",
                        date=current_date,
                        calendar_type=CalendarType.SOLAR,
                        description={
                            "en": f"Potential solar eclipse on {current_date}",
                            "zh": f"{current_date} 可能發生日食",
                            "zh-tw": f"{current_date} 可能發生日食"
                        },
                        astronomical_significance="Sun-Moon-Earth alignment"
                    )
                    events.append(eclipse_event)
            
            # 簡化的月食預測 (Simplified lunar eclipse prediction)
            elif lunar_phase["phase_name"] == "full_moon":
                # 月食可能發生在滿月時 (Lunar eclipses occur during full moon)
                if abs(lunar_phase["days_since_new_moon"] - self.astro_data.lunar_month/2) < 0.5:
                    eclipse_event = CalendarEvent(
                        name="lunar_eclipse", 
                        date=current_date,
                        calendar_type=CalendarType.LUNAR,
                        description={
                            "en": f"Potential lunar eclipse on {current_date}",
                            "zh": f"{current_date} 可能發生月食",
                            "zh-tw": f"{current_date} 可能發生月食"
                        },
                        astronomical_significance="Earth-Moon-Sun alignment"
                    )
                    events.append(eclipse_event)
            
            current_date += datetime.timedelta(days=1)
        
        return events
    
    def solar_terms_calculation(self, year: int) -> List[CalendarEvent]:
        """計算二十四節氣 (Calculate 24 Solar Terms)"""
        solar_terms = [
            ("立春", "Beginning of Spring"), ("雨水", "Rain Water"),
            ("驚蟄", "Awakening of Insects"), ("春分", "Vernal Equinox"),
            ("清明", "Clear and Bright"), ("穀雨", "Grain Rain"),
            ("立夏", "Beginning of Summer"), ("小滿", "Grain Buds"),
            ("芒種", "Grain in Ear"), ("夏至", "Summer Solstice"),
            ("小暑", "Slight Heat"), ("大暑", "Great Heat"),
            ("立秋", "Beginning of Autumn"), ("處暑", "Stopping the Heat"),
            ("白露", "White Dew"), ("秋分", "Autumnal Equinox"),
            ("寒露", "Cold Dew"), ("霜降", "Frost's Descent"),
            ("立冬", "Beginning of Winter"), ("小雪", "Slight Snow"),
            ("大雪", "Great Snow"), ("冬至", "Winter Solstice"),
            ("小寒", "Slight Cold"), ("大寒", "Great Cold")
        ]
        
        events = []
        # 每個節氣約15.2天 (Each solar term is approximately 15.2 days)
        days_per_term = 365.25 / 24
        
        for i, (chinese_name, english_name) in enumerate(solar_terms):
            # 立春約在2月4日 (Beginning of Spring around February 4th)
            start_of_year = datetime.date(year, 2, 4)
            term_date = start_of_year + datetime.timedelta(days=i * days_per_term)
            
            event = CalendarEvent(
                name=f"solar_term_{i+1}",
                date=term_date,
                calendar_type=CalendarType.SOLAR,
                description={
                    "en": english_name,
                    "zh": chinese_name,
                    "zh-tw": chinese_name
                },
                astronomical_significance=f"Solar longitude {15*i}°"
            )
            events.append(event)
        
        return events
    
    def calendar_conversion_table(self, date: datetime.date) -> Dict[str, Dict[str, Union[str, int]]]:
        """多曆法轉換表 (Multi-calendar conversion table)"""
        lunar_phase = self.lunar_phase_calculation(date)
        chinese_cal = self.chinese_calendar_conversion(date)
        
        # 伊斯蘭曆簡化計算 (Simplified Islamic calendar)
        islamic_epoch = datetime.date(622, 7, 16)  # 伊斯蘭紀元
        days_since_islamic_epoch = (date - islamic_epoch).days
        islamic_year = int(days_since_islamic_epoch / (12 * self.astro_data.lunar_month)) + 1
        
        # 希伯來曆簡化計算 (Simplified Hebrew calendar)
        hebrew_year = date.year + 3760  # 希伯來紀年
        
        return {
            "gregorian": {
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "weekday": calendar.day_name[date.weekday()]
            },
            "chinese_lunar": chinese_cal,
            "islamic": {
                "year": islamic_year,
                "month": int((days_since_islamic_epoch % (12 * self.astro_data.lunar_month)) / self.astro_data.lunar_month) + 1,
                "day": int(days_since_islamic_epoch % self.astro_data.lunar_month) + 1
            },
            "hebrew": {
                "year": hebrew_year,
                "month": date.month,  # 簡化處理
                "day": date.day
            },
            "lunar_phase": lunar_phase,
            "astronomical": {
                "julian_day": self.solar_calendar_days_since_epoch(date) + 2451545.0,  # J2000.0 = JD 2451545.0
                "days_since_j2000": self.solar_calendar_days_since_epoch(date)
            }
        }
    
    def generate_multilingual_report(self, date: datetime.date, language: str = "en") -> str:
        """生成多語言曆法報告 (Generate multilingual calendar report)"""
        conversion_table = self.calendar_conversion_table(date)
        
        # 標題 (Title)
        titles = {
            "en": f"Calendar Report for {date}",
            "zh": f"{date} 曆法報告",
            "zh-tw": f"{date} 曆法報告"
        }
        title = titles.get(language, titles["en"])
        
        report = f"{'='*50}\n{title}\n{'='*50}\n\n"
        
        # 基本信息 (Basic Information)
        basic_info = {
            "en": "Basic Calendar Information:",
            "zh": "基本曆法信息：",
            "zh-tw": "基本曆法信息："
        }
        report += f"{basic_info.get(language, basic_info['en'])}\n"
        
        # 格里高利曆 (Gregorian Calendar)
        gregorian = conversion_table["gregorian"]
        report += f"- {self.get_term('solar_calendar', language)}: {gregorian['year']}-{gregorian['month']:02d}-{gregorian['day']:02d} ({gregorian['weekday']})\n"
        
        # 中國農曆 (Chinese Lunar Calendar)
        chinese = conversion_table["chinese_lunar"]
        report += f"- {self.get_term('lunar_calendar', language)}: {chinese['lunar_year']}年{chinese['lunar_month']}月{chinese['lunar_day']}日 ({chinese['year_stem_branch']}年)\n"
        
        # 月相信息 (Lunar Phase Information)
        lunar_phase = conversion_table["lunar_phase"]
        phase_info = {
            "en": "Lunar Phase Information:",
            "zh": "月相信息：",
            "zh-tw": "月相信息："
        }
        report += f"\n{phase_info.get(language, phase_info['en'])}\n"
        report += f"- {self.get_term(lunar_phase['phase_name'], language)}\n"
        report += f"- 照明比例 (Illumination): {lunar_phase['illumination_fraction']:.2%}\n"
        report += f"- 月相角度 (Phase Angle): {lunar_phase['phase_angle']:.1f}°\n"
        
        # 天文信息 (Astronomical Information)
        astro_info = {
            "en": "Astronomical Information:",
            "zh": "天文信息：", 
            "zh-tw": "天文信息："
        }
        report += f"\n{astro_info.get(language, astro_info['en'])}\n"
        astronomical = conversion_table["astronomical"]
        report += f"- 儒略日 (Julian Day): {astronomical['julian_day']:.1f}\n"
        report += f"- J2000.0起算天數 (Days since J2000.0): {astronomical['days_since_j2000']:.1f}\n"
        
        return report


def demonstrate_solar_lunar_calendar_3body():
    """演示日月陽曆三體系統功能 (Demonstrate Solar Lunar Calendar 3-Body System functionality)"""
    system = SolarLunarCalendar3Body()
    
    print("日月陽曆(3體)系統演示 / Solar Lunar Calendar Three-Body System Demonstration")
    print("="*80)
    
    # 當前日期 (Current date)
    today = datetime.date.today()
    
    # 生成多語言報告 (Generate multilingual reports)
    print("\n1. 多語言曆法報告 (Multilingual Calendar Reports):")
    print("-" * 50)
    
    for lang in ["en", "zh", "zh-tw"]:
        print(f"\n{lang.upper()}:")
        print(system.generate_multilingual_report(today, lang))
    
    # 月相計算 (Lunar phase calculation)
    print("\n2. 月相計算 (Lunar Phase Calculations):")
    print("-" * 50)
    lunar_phase = system.lunar_phase_calculation(today)
    for key, value in lunar_phase.items():
        print(f"{key}: {value}")
    
    # 三體位置計算 (Three-body position calculation)
    print("\n3. 三體位置計算 (Three-Body Position Calculations):")
    print("-" * 50)
    days_since_epoch = system.solar_calendar_days_since_epoch(today)
    positions = system.three_body_position(days_since_epoch)
    
    print(f"太陽位置 (Sun Position): {positions.sun_pos}")
    print(f"地球位置 (Earth Position): {positions.earth_pos}")
    print(f"月球位置 (Moon Position): {positions.moon_pos}")
    
    # 引力計算 (Gravitational force calculation)
    print("\n4. 三體引力計算 (Three-Body Gravitational Forces):")
    print("-" * 50)
    forces = system.three_body_forces(positions)
    for body, force in forces.items():
        print(f"{body.capitalize()} force: {force}")
    
    # 日食月食預測 (Eclipse prediction)
    print("\n5. 日食月食預測 (Eclipse Predictions):")
    print("-" * 50)
    end_date = today + datetime.timedelta(days=365)
    eclipses = system.eclipse_prediction(today, end_date)
    
    for eclipse in eclipses[:5]:  # 顯示前5個預測
        print(f"{eclipse.date}: {eclipse.description['zh']} / {eclipse.description['en']}")
    
    # 二十四節氣 (24 Solar Terms)
    print("\n6. 二十四節氣 (24 Solar Terms):")
    print("-" * 50)
    solar_terms = system.solar_terms_calculation(today.year)
    
    for term in solar_terms[:6]:  # 顯示前6個節氣
        print(f"{term.date}: {term.description['zh']} / {term.description['en']}")
    
    # 曆法術語演示 (Calendar terminology demonstration)
    print("\n7. 多語言曆法術語 (Multilingual Calendar Terminology):")
    print("-" * 50)
    terms = ["solar_calendar", "lunar_calendar", "three_body_system", "new_moon", "full_moon"]
    languages = ["en", "zh", "ja", "ko", "fr", "de", "es", "ru", "ar"]
    
    for term in terms:
        print(f"\n{term}:")
        for lang in languages:
            translation = system.get_term(term, lang)
            print(f"  {lang}: {translation}")


if __name__ == "__main__":
    demonstrate_solar_lunar_calendar_3body()
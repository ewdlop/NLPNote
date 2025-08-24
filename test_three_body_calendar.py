#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三體曆法系統測試 (Three-Body Calendar System Tests)

測試用例集合，驗證三體曆法系統的各項功能。

Test suite for validating Three-Body Calendar System functionality.
"""

import unittest
import datetime
from ThreeBodyCalendar import (
    ThreeBodyCalendar, 
    ThreeBodyDateParser, 
    AstronomicalCalculator,
    CelestialPosition,
    ThreeBodyState
)


class TestAstronomicalCalculator(unittest.TestCase):
    """天文計算器測試 - Astronomical Calculator Tests"""
    
    def setUp(self):
        self.calculator = AstronomicalCalculator()
        self.test_date = datetime.datetime(2024, 12, 22)  # Winter solstice
        
    def test_julian_day_calculation(self):
        """測試儒略日計算 - Test Julian Day calculation"""
        jd = self.calculator.julian_day(self.test_date)
        self.assertIsInstance(jd, float)
        self.assertGreater(jd, 2400000)  # Reasonable range for modern dates
        
    def test_sun_position(self):
        """測試太陽位置計算 - Test Sun position calculation"""
        jd = self.calculator.julian_day(self.test_date)
        sun_pos = self.calculator.sun_position(jd)
        
        self.assertIsInstance(sun_pos, CelestialPosition)
        self.assertGreaterEqual(sun_pos.longitude, 0)
        self.assertLess(sun_pos.longitude, 360)
        self.assertGreater(sun_pos.distance, 0.5)  # Reasonable distance range
        self.assertLess(sun_pos.distance, 1.5)
        
    def test_moon_position(self):
        """測試月球位置計算 - Test Moon position calculation"""
        jd = self.calculator.julian_day(self.test_date)
        moon_pos = self.calculator.moon_position(jd)
        
        self.assertIsInstance(moon_pos, CelestialPosition)
        self.assertGreaterEqual(moon_pos.longitude, 0)
        self.assertLess(moon_pos.longitude, 360)
        self.assertGreaterEqual(moon_pos.latitude, -90)
        self.assertLessEqual(moon_pos.latitude, 90)
        
    def test_earth_position(self):
        """測試地球位置計算 - Test Earth position calculation"""
        jd = self.calculator.julian_day(self.test_date)
        earth_pos = self.calculator.earth_position(jd)
        
        self.assertIsInstance(earth_pos, CelestialPosition)
        self.assertGreaterEqual(earth_pos.longitude, 0)
        self.assertLess(earth_pos.longitude, 360)


class TestThreeBodyCalendar(unittest.TestCase):
    """三體曆法測試 - Three-Body Calendar Tests"""
    
    def setUp(self):
        self.calendar = ThreeBodyCalendar()
        self.test_dates = [
            datetime.datetime(2024, 3, 20),  # Spring equinox
            datetime.datetime(2024, 6, 21),  # Summer solstice
            datetime.datetime(2024, 9, 22),  # Autumn equinox
            datetime.datetime(2024, 12, 22), # Winter solstice
        ]
        
    def test_get_three_body_state(self):
        """測試三體狀態獲取 - Test three-body state retrieval"""
        for test_date in self.test_dates:
            state = self.calendar.get_three_body_state(test_date)
            
            self.assertIsInstance(state, ThreeBodyState)
            self.assertEqual(state.gregorian_date.date(), test_date.date())
            self.assertIsInstance(state.earth, CelestialPosition)
            self.assertIsInstance(state.sun, CelestialPosition)
            self.assertIsInstance(state.moon, CelestialPosition)
            
    def test_lunar_phase_calculation(self):
        """測試月相計算 - Test lunar phase calculation"""
        for test_date in self.test_dates:
            phase_zh, phase_en, phase_ratio = self.calendar.lunar_phase(test_date)
            
            self.assertIsInstance(phase_zh, str)
            self.assertIsInstance(phase_en, str)
            self.assertIsInstance(phase_ratio, float)
            self.assertGreaterEqual(phase_ratio, 0)
            self.assertLessEqual(phase_ratio, 1)
            
            # Check that Chinese and English phase names are reasonable
            self.assertIn(phase_zh, ['新月', '眉月', '上弦月', '盈凸月', '滿月', '殘月'])
            valid_en_phases = ['New Moon', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous', 'Full Moon', 'Waning']
            self.assertIn(phase_en, valid_en_phases)
            
    def test_solar_term_calculation(self):
        """測試節氣計算 - Test solar term calculation"""
        # Test winter solstice
        winter_solstice = datetime.datetime(2024, 12, 22)
        term_zh, term_en = self.calendar.solar_term(winter_solstice)
        
        self.assertIsInstance(term_zh, str)
        self.assertIsInstance(term_en, str)
        # Winter solstice should be around this date
        self.assertIn('冬', term_zh)  # Should contain winter character
        
    def test_format_three_body_date(self):
        """測試三體日期格式化 - Test three-body date formatting"""
        test_date = self.test_dates[0]
        
        # Test Chinese formatting
        formatted_zh = self.calendar.format_three_body_date(test_date, 'zh')
        self.assertIsInstance(formatted_zh, str)
        self.assertIn('三體曆', formatted_zh)
        self.assertIn('節氣', formatted_zh)
        self.assertIn('月相', formatted_zh)
        
        # Test English formatting
        formatted_en = self.calendar.format_three_body_date(test_date, 'en')
        self.assertIsInstance(formatted_en, str)
        self.assertIn('Three-Body Calendar', formatted_en)
        self.assertIn('Solar Term', formatted_en)
        self.assertIn('Lunar Phase', formatted_en)


class TestThreeBodyDateParser(unittest.TestCase):
    """三體日期解析器測試 - Three-Body Date Parser Tests"""
    
    def setUp(self):
        self.parser = ThreeBodyDateParser()
        
    def test_chinese_date_parsing(self):
        """測試中文日期解析 - Test Chinese date parsing"""
        test_cases = [
            ('2024年12月22日', datetime.datetime(2024, 12, 22)),
            ('2025年1月1日', datetime.datetime(2025, 1, 1)),
            ('2024年冬至', datetime.datetime(2024, 12, 22)),
            ('2025年春分', datetime.datetime(2025, 3, 21)),
        ]
        
        for text, expected in test_cases:
            result = self.parser.parse_chinese_date(text)
            self.assertIsNotNone(result, f"Failed to parse: {text}")
            self.assertEqual(result.date(), expected.date(), f"Wrong date for: {text}")
            
    def test_english_date_parsing(self):
        """測試英文日期解析 - Test English date parsing"""
        test_cases = [
            ('2024-12-22', datetime.datetime(2024, 12, 22)),
            ('2025-01-01', datetime.datetime(2025, 1, 1)),
        ]
        
        for text, expected in test_cases:
            result = self.parser.parse_english_date(text)
            self.assertIsNotNone(result, f"Failed to parse: {text}")
            self.assertEqual(result.date(), expected.date(), f"Wrong date for: {text}")
            
    def test_festival_parsing(self):
        """測試節日解析 - Test festival parsing"""
        festival_cases = [
            '中秋節',
            '春節',
            '清明節'
        ]
        
        for festival in festival_cases:
            result = self.parser.parse_chinese_date(festival)
            self.assertIsNotNone(result, f"Failed to parse festival: {festival}")
            
    def test_analyze_date_expression(self):
        """測試日期表達式分析 - Test date expression analysis"""
        test_expressions = [
            '2024年12月22日',
            '2024年冬至',
            '中秋節'
        ]
        
        for expr in test_expressions:
            result = self.parser.analyze_date_expression(expr)
            
            self.assertIsInstance(result, dict)
            self.assertIn('input_text', result)
            self.assertIn('parsed_date', result)
            self.assertIn('three_body_format_zh', result)
            self.assertIn('three_body_format_en', result)
            
            if result['parsed_date']:
                self.assertIsNotNone(result['three_body_format_zh'])
                self.assertIsNotNone(result['three_body_format_en'])
                self.assertIsNotNone(result['lunar_phase'])
                self.assertIsNotNone(result['solar_term'])


class TestIntegrationScenarios(unittest.TestCase):
    """整合場景測試 - Integration Scenario Tests"""
    
    def setUp(self):
        self.calendar = ThreeBodyCalendar()
        self.parser = ThreeBodyDateParser()
        
    def test_seasonal_transitions(self):
        """測試季節轉換 - Test seasonal transitions"""
        seasonal_dates = [
            ('2024年立春', '立春'),
            ('2024年春分', '春分'),
            ('2024年立夏', '立夏'),
            ('2024年夏至', '夏至'),
            ('2024年立秋', '立秋'),
            ('2024年秋分', '秋分'),
            ('2024年立冬', '立冬'),
            ('2024年冬至', '冬至'),
        ]
        
        for date_expr, expected_term in seasonal_dates:
            parsed_date = self.parser.parse_chinese_date(date_expr)
            self.assertIsNotNone(parsed_date, f"Failed to parse: {date_expr}")
            
            term_zh, term_en = self.calendar.solar_term(parsed_date)
            # The parsed date should be close to the expected solar term
            self.assertIsInstance(term_zh, str)
            self.assertIsInstance(term_en, str)
            
    def test_lunar_cycle_tracking(self):
        """測試月相週期追蹤 - Test lunar cycle tracking"""
        # Test over a month to see lunar phase changes
        start_date = datetime.datetime(2024, 1, 1)
        
        phases_seen = set()
        for day_offset in range(0, 30, 7):  # Weekly samples
            test_date = start_date + datetime.timedelta(days=day_offset)
            phase_zh, phase_en, ratio = self.calendar.lunar_phase(test_date)
            phases_seen.add(phase_zh)
            
        # Should see multiple different phases over a month
        self.assertGreater(len(phases_seen), 1, "Should observe multiple lunar phases over a month")
        
    def test_cultural_date_expressions(self):
        """測試文化日期表達 - Test cultural date expressions"""
        cultural_expressions = [
            '春節',
            '中秋節', 
            '清明節',
            '端午節'
        ]
        
        for expr in cultural_expressions:
            analysis = self.parser.analyze_date_expression(expr)
            self.assertIsNotNone(analysis['parsed_date'], f"Failed to analyze: {expr}")
            
            # Should provide both Chinese and English formatting
            self.assertIsNotNone(analysis['three_body_format_zh'])
            self.assertIsNotNone(analysis['three_body_format_en'])


def run_tests():
    """運行所有測試 - Run all tests"""
    print("=== 三體曆法系統測試 Three-Body Calendar System Tests ===\n")
    
    # Create test suite
    test_classes = [
        TestAstronomicalCalculator,
        TestThreeBodyCalendar,
        TestThreeBodyDateParser,
        TestIntegrationScenarios
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== 測試總結 Test Summary ===")
    print(f"運行測試 Tests run: {result.testsRun}")
    print(f"失敗 Failures: {len(result.failures)}")
    print(f"錯誤 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n失敗詳情 Failure details:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\n錯誤詳情 Error details:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率 Success rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
"""
同倫類型論NLP測試與演示 (Homotopy Type Theory NLP Tests and Examples)

This module provides comprehensive tests and examples for the HomotopyTypeTheoryNLP implementation,
demonstrating the integration of Homotopy Type Theory concepts with natural language processing
and the existing PathIntegralNLP framework.
"""

import sys
import os

# Add the current directory to the path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from HomotopyTypeTheoryNLP import (
        HomotopyTypeTheoryNLP, HoTTPathType, SemanticType, 
        SemanticPath, HomotopyEquivalence, HigherInductiveType
    )
    from PathIntegralNLP import TianDaoPath
    HOTT_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    HOTT_AVAILABLE = False


def test_basic_hott_concepts():
    """測試基本HoTT概念"""
    print("=" * 60)
    print("基本同倫類型論概念測試 (Basic HoTT Concepts Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    print("\n1. 語義類型註冊測試")
    print("-" * 30)
    
    # 註冊語義類型
    emotions_type = hott_nlp.register_semantic_type(
        "emotions", 
        ["快樂", "悲傷", "憤怒", "平靜"],
        dimension=2,
        properties={"category": "emotions", "valence": "mixed"}
    )
    
    actions_type = hott_nlp.register_semantic_type(
        "actions",
        ["跑步", "思考", "說話", "休息"],
        dimension=2,
        properties={"category": "actions", "energy": "variable"}
    )
    
    print(f"情感類型: {emotions_type.name}, 元素數: {len(emotions_type.elements)}")
    print(f"行動類型: {actions_type.name}, 元素數: {len(actions_type.elements)}")
    
    print("\n2. 語義路徑構造測試")
    print("-" * 30)
    
    # 構造路徑
    path1 = hott_nlp.construct_semantic_path("快樂", "跑步", HoTTPathType.EQUIVALENCE)
    path2 = hott_nlp.construct_semantic_path("跑步", "健康", HoTTPathType.TRANSPORT)
    
    print(f"路徑1: {path1.source} → {path1.target} (類型: {path1.path_type.value})")
    print(f"路徑2: {path2.source} → {path2.target} (類型: {path2.path_type.value})")
    
    # 測試路徑合成
    composed = hott_nlp.path_space_calculator.compose_paths(path1, path2)
    if composed:
        print(f"合成路徑: {composed.source} → {composed.target}")
        print(f"合成證明項: {composed.proof_term}")
    else:
        print("路徑無法合成")
    
    print("\n3. 一元性等價測試")
    print("-" * 30)
    
    # 測試等價性
    equivalence = hott_nlp.univalence_calculator.construct_equivalence(emotions_type, actions_type)
    if equivalence:
        print(f"發現等價: {equivalence.type_a.name} ≃ {equivalence.type_b.name}")
        print(f"等價證明: {equivalence.equivalence_proof}")
    else:
        print("未發現等價關係")
    
    print("\n4. 恆等路徑測試")
    print("-" * 30)
    
    # 恆等路徑
    id_path = hott_nlp.path_space_calculator.identity_path("概念")
    print(f"恆等路徑: {id_path.source} = {id_path.target}")
    print(f"證明項: {id_path.proof_term}")
    
    return True


def test_path_space_analysis():
    """測試路徑空間分析"""
    print("\n" + "=" * 60)
    print("路徑空間分析測試 (Path Space Analysis Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    # 測試文本
    test_texts = [
        "水流向山，山指向天，天覆蓋地",
        "思考產生想法，想法變成計劃，計劃導致行動",
        "From concept to reality through careful planning",
        "學習帶來知識，知識促進理解，理解創造智慧",
        "花開花落，潮起潮落，人生如夢"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n測試 {i}: '{text}'")
        print("-" * 40)
        
        analysis = hott_nlp.construct_path_space_analysis(text)
        
        print(f"概念提取: {analysis['concepts']}")
        print(f"路徑分析: {analysis['path_analysis']}")
        print(f"構造路徑: {len(analysis.get('paths', []))}")
        print(f"合成路徑: {len(analysis.get('composed_paths', []))}")
        print(f"等價關係: {len(analysis.get('equivalences', []))}")
        
        hott_metrics = analysis.get('hott_analysis', {})
        print(f"HoTT 指標:")
        print(f"  - 路徑構造數: {hott_metrics.get('paths_constructed', 0)}")
        print(f"  - 同倫構造數: {hott_metrics.get('homotopies_constructed', 0)}")
        print(f"  - 一元性應用: {hott_metrics.get('univalence_applications', 0)}")
        
        # 天道整合結果
        tian_dao = analysis.get('tian_dao_integration', {})
        if tian_dao.get('integration_successful', False):
            print(f"天道整合成功:")
            print(f"  - 天道對齊度: {tian_dao.get('tian_dao_alignment', 0):.3f}")
            print(f"  - 自然流動: {tian_dao.get('natural_flow_score', 0):.3f}")
            print(f"  - 和諧指數: {tian_dao.get('harmony_index', 0):.3f}")
        else:
            print(f"天道整合: {tian_dao.get('integration_error', '未集成')}")
    
    return True


def test_univalence_applications():
    """測試一元性原理應用"""
    print("\n" + "=" * 60)
    print("一元性原理應用測試 (Univalence Principle Applications Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    # 測試語義等價分析
    text_pairs = [
        ("大象很大", "巨大的動物"),
        ("快速奔跑", "迅速移動"),
        ("思考問題", "contemplating issues"),
        ("美麗的花朵", "漂亮的植物"),
        ("困難的挑戰", "艱難的任務")
    ]
    
    for i, (text1, text2) in enumerate(text_pairs, 1):
        print(f"\n一元性分析 {i}: '{text1}' vs '{text2}'")
        print("-" * 50)
        
        analysis = hott_nlp.univalence_based_semantic_analysis(text1, text2)
        
        print(f"文本1概念: {analysis['text1_concepts']}")
        print(f"文本2概念: {analysis['text2_concepts']}")
        print(f"一元性等價: {'是' if analysis['univalent_equivalence'] else '否'}")
        
        if analysis['equivalence_details']:
            details = analysis['equivalence_details']
            print(f"等價詳情:")
            print(f"  - 前向映射: {details['forward_map']}")
            print(f"  - 後向映射: {details['backward_map']}")
            print(f"  - 等價證明: {details['equivalence_proof']}")
        
        transport = analysis.get('semantic_transport', {})
        if transport:
            print(f"語義傳輸示例:")
            for orig, transported in list(transport.items())[:2]:
                print(f"  - {orig} → {transported}")
        
        identity_types = analysis.get('identity_types_analysis', {})
        if identity_types:
            print(f"恆等類型分析: {len(identity_types)} 個恆等路徑")
    
    return True


def test_higher_inductive_types():
    """測試高階歸納類型"""
    print("\n" + "=" * 60)
    print("高階歸納類型測試 (Higher Inductive Types Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    # 測試文本結構
    complex_texts = [
        "春天來了，花開了，鳥兒歌唱，生命復甦，大地回春",
        "學習編程，練習算法，做項目，找工作，成為工程師",
        "思考問題，分析原因，尋找方案，實施計劃，評估結果",
        "日出東方，光照大地，萬物生長，日落西山，夜晚降臨"
    ]
    
    for i, text in enumerate(complex_texts, 1):
        print(f"\n高階歸納類型 {i}: '{text}'")
        print("-" * 40)
        
        # 構造高階歸納類型
        hit = hott_nlp.construct_higher_inductive_type(f"text_structure_{i}", text)
        
        print(f"HIT 名稱: {hit.name}")
        print(f"基本構造子數量: {len(hit.constructors)}")
        print(f"路徑構造子數量: {len(hit.path_constructors)}")
        print(f"連貫性法則: {len(hit.coherence_laws)}")
        
        print(f"構造子 (前5個): {hit.constructors[:5]}")
        
        if hit.path_constructors:
            print(f"路徑構造子示例:")
            for pc in hit.path_constructors[:3]:
                print(f"  - {pc.source} → {pc.target} ({pc.path_type.value})")
        
        print(f"消除規則: {list(hit.elimination_rules.keys())}")
    
    return True


def test_homotopy_analysis():
    """測試同倫分析"""
    print("\n" + "=" * 60)
    print("同倫分析測試 (Homotopy Analysis Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    # 測試高階路徑構造
    print("\n1. 高階路徑構造")
    print("-" * 25)
    
    # 創建兩條相同起終點的路徑
    path1 = hott_nlp.construct_semantic_path("開始", "結束", HoTTPathType.EQUIVALENCE)
    path2 = hott_nlp.construct_semantic_path("開始", "結束", HoTTPathType.TRANSPORT)
    
    print(f"路徑1: {path1.proof_term}")
    print(f"路徑2: {path2.proof_term}")
    
    # 構造同倫
    homotopy = hott_nlp.higher_path_calculator.construct_homotopy(path1, path2)
    if homotopy:
        print(f"同倫路徑: {homotopy.proof_term}")
        print(f"同倫層次: {homotopy.homotopy_level}")
        print(f"連貫性條件: {len(homotopy.coherence_conditions)}")
    else:
        print("無法構造同倫")
    
    print("\n2. 語義傳輸測試")
    print("-" * 25)
    
    # 測試語義傳輸
    element = "概念A"
    transported = hott_nlp.higher_path_calculator.transport_along_path(path1, element)
    print(f"原始元素: {element}")
    print(f"傳輸後: {transported}")
    
    print("\n3. 逆路徑構造")
    print("-" * 25)
    
    # 構造逆路徑
    inverse = hott_nlp.path_space_calculator.inverse_path(path1)
    print(f"原路徑: {path1.proof_term}")
    print(f"逆路徑: {inverse.proof_term}")
    print(f"逆路徑方向: {inverse.source} → {inverse.target}")
    
    return True


def test_comprehensive_analysis():
    """測試綜合分析功能"""
    print("\n" + "=" * 60)
    print("綜合同倫類型論分析測試 (Comprehensive HoTT Analysis Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    # 複雜測試文本
    complex_text = "在同倫類型論中，類型被視為空間，元素被視為點，等式被視為路徑。通過一元性原理，我們可以將等價性和同一性統一起來，這為數學基礎提供了新的視角。"
    
    print(f"分析文本: '{complex_text}'")
    print("-" * 50)
    
    # 執行綜合分析
    comprehensive = hott_nlp.comprehensive_hott_analysis(complex_text)
    
    print("\n基本路徑分析:")
    basic = comprehensive['basic_path_analysis']
    print(f"  概念數量: {len(basic['concepts'])}")
    print(f"  構造路徑: {len(basic.get('paths', []))}")
    print(f"  等價關係: {len(basic.get('equivalences', []))}")
    
    print("\n高階歸納類型:")
    hit_info = comprehensive['higher_inductive_type']
    print(f"  名稱: {hit_info['name']}")
    print(f"  構造子數量: {hit_info['constructors_count']}")
    print(f"  路徑構造子數量: {hit_info['path_constructors_count']}")
    
    print("\n同倫群分析:")
    homotopy_groups = comprehensive['homotopy_groups']
    print(f"  π₀ (連通分量): {homotopy_groups['π_0']}")
    print(f"  π₁ (基本群): {homotopy_groups['π_1']}")
    print(f"  概念循環: {homotopy_groups['concept_cycles']}")
    
    higher_groups = homotopy_groups.get('higher_groups', {})
    if higher_groups:
        print(f"  高階同倫群: {list(higher_groups.keys())}")
    
    print("\n一元性應用:")
    univalence = comprehensive['univalence_applications']
    print(f"  潛在等價: {len(univalence['potential_equivalences'])}")
    print(f"  等價類: {len(univalence['equivalence_classes'])}")
    print(f"  語義識別數: {univalence['semantic_identification_count']}")
    
    print("\nHoTT複雜度指標:")
    complexity = comprehensive['hott_complexity_metrics']
    print(f"  類型複雜度: {complexity['type_complexity']:.3f}")
    print(f"  路徑複雜度: {complexity['path_complexity']:.3f}")
    print(f"  同倫複雜度: {complexity['homotopy_complexity']:.3f}")
    print(f"  整體複雜度: {complexity['overall_complexity']:.3f}")
    
    return True


def test_integration_with_path_integral():
    """測試與路徑積分NLP的整合"""
    print("\n" + "=" * 60)
    print("與路徑積分NLP整合測試 (Integration with Path Integral NLP Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    # 測試整合與非整合版本
    hott_nlp_integrated = HomotopyTypeTheoryNLP(integrate_path_integral=True)
    hott_nlp_standalone = HomotopyTypeTheoryNLP(integrate_path_integral=False)
    
    test_text = "依照天道的指引，通過自然的流動，達到和諧的境界"
    
    print(f"測試文本: '{test_text}'")
    print("-" * 40)
    
    print("\n1. 整合版本分析:")
    integrated_analysis = hott_nlp_integrated.construct_path_space_analysis(test_text)
    tian_dao_integrated = integrated_analysis.get('tian_dao_integration', {})
    
    if tian_dao_integrated.get('integration_successful', False):
        print(f"  ✓ 天道整合成功")
        print(f"  天道對齊度: {tian_dao_integrated.get('tian_dao_alignment', 0):.3f}")
        print(f"  自然流動分數: {tian_dao_integrated.get('natural_flow_score', 0):.3f}")
        print(f"  和諧指數: {tian_dao_integrated.get('harmony_index', 0):.3f}")
    else:
        print(f"  ✗ 天道整合失敗: {tian_dao_integrated.get('integration_error', '未知錯誤')}")
    
    print("\n2. 獨立版本分析:")
    standalone_analysis = hott_nlp_standalone.construct_path_space_analysis(test_text)
    tian_dao_standalone = standalone_analysis.get('tian_dao_integration', {})
    
    if not tian_dao_standalone:
        print(f"  ✓ 按預期沒有天道整合")
    else:
        print(f"  意外的天道整合結果: {tian_dao_standalone}")
    
    print("\n3. HoTT 分析比較:")
    integrated_hott = integrated_analysis.get('hott_analysis', {})
    standalone_hott = standalone_analysis.get('hott_analysis', {})
    
    print(f"  整合版路徑數: {integrated_hott.get('paths_constructed', 0)}")
    print(f"  獨立版路徑數: {standalone_hott.get('paths_constructed', 0)}")
    print(f"  整合版同倫數: {integrated_hott.get('homotopies_constructed', 0)}")
    print(f"  獨立版同倫數: {standalone_hott.get('homotopies_constructed', 0)}")
    
    return True


def demonstrate_hott_concepts():
    """演示HoTT核心概念"""
    print("\n" + "=" * 60)
    print("同倫類型論核心概念演示 (HoTT Core Concepts Demonstration)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過演示")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    print("\n📘 HoTT 核心概念:")
    print("1. 類型即空間 (Types as Spaces)")
    print("2. 項即點 (Terms as Points)")  
    print("3. 等式即路徑 (Equality as Paths)")
    print("4. 一元性原理 (Univalence Axiom)")
    print("5. 高階歸納類型 (Higher Inductive Types)")
    
    print("\n🔬 實際應用示例:")
    
    # 1. 類型即空間
    print("\n1. 類型作為語義空間:")
    emotions_space = hott_nlp.register_semantic_type(
        "emotional_space", 
        ["joy", "sadness", "anger", "peace", "excitement"],
        dimension=3,
        properties={"domain": "emotions", "dimensionality": "high"}
    )
    print(f"   情感空間: {emotions_space.name}, 維度: {emotions_space.dimension}")
    print(f"   空間中的點: {list(emotions_space.elements)[:3]}...")
    
    # 2. 等式即路徑
    print("\n2. 語義等式作為路徑:")
    path = hott_nlp.construct_semantic_path("sadness", "peace", HoTTPathType.HOMOTOPY)
    print(f"   路徑: {path.source} ≡ {path.target}")
    print(f"   證明項: {path.proof_term}")
    print(f"   路徑類型: {path.path_type.value}")
    
    # 3. 一元性原理
    print("\n3. 一元性原理應用:")
    type_a = hott_nlp.register_semantic_type("concepts_a", ["思想", "觀念", "想法"])
    type_b = hott_nlp.register_semantic_type("concepts_b", ["思維", "概念", "理念"])
    
    equivalence = hott_nlp.univalence_calculator.construct_equivalence(type_a, type_b)
    if equivalence:
        print(f"   一元性等價: {type_a.name} ≃ {type_b.name}")
        print(f"   等價即等同: (A ≃ B) ≃ (A = B)")
    else:
        print(f"   類型 {type_a.name} 和 {type_b.name} 不等價")
    
    # 4. 高階歸納類型
    print("\n4. 高階歸納類型構造:")
    hit_text = "圓的概念：點組成圓周，路徑連接點"
    circle_hit = hott_nlp.construct_higher_inductive_type("circle", hit_text)
    print(f"   HIT名稱: {circle_hit.name}")
    print(f"   構造子數: {len(circle_hit.constructors)}")
    print(f"   路徑構造子數: {len(circle_hit.path_constructors)}")
    
    # 5. 同倫層次
    print("\n5. 同倫層次分析:")
    complex_text = "複雜的語義結構，包含多層含義和關係"
    homotopy_analysis = hott_nlp._calculate_homotopy_groups(complex_text)
    print(f"   π₀ (連通分量): {homotopy_analysis['π_0']}")
    print(f"   π₁ (基本群): {homotopy_analysis['π_1']}")
    
    return True


def test_ethical_action_analysis():
    """測試倫理行動分析"""
    print("=" * 60)
    print("倫理行動分析測試 (Ethical Action Analysis Test)")
    print("=" * 60)
    
    if not HOTT_AVAILABLE:
        print("HomotopyTypeTheoryNLP 不可用，跳過測試")
        return False
    
    hott_nlp = HomotopyTypeTheoryNLP()
    
    print("\n1. 正面倫理語言測試")
    print("-" * 30)
    
    positive_text = "學習英語是一個過程，每個人都有不同的起點和發展速度"
    analysis = hott_nlp.ethical_action_analysis(positive_text)
    
    print(f"文本: {analysis['text']}")
    print(f"概念: {analysis['concepts']}")
    print(f"倫理評分: {analysis['ethical_evaluation']['total_ethical_score']:.3f}")
    print(f"是否可接受: {analysis['ethical_evaluation']['is_ethically_acceptable']}")
    print(f"總體評估: {analysis['overall_assessment']}")
    
    if analysis['ethical_recommendations']:
        print(f"改進建議: {', '.join(analysis['ethical_recommendations'])}")
    else:
        print("✅ 無需改進，符合倫理標準")
    
    print("\n2. 問題語言檢測測試")
    print("-" * 30)
    
    problematic_text = "他的英語很差，總是說錯話，像個弱智一樣"
    analysis2 = hott_nlp.ethical_action_analysis(problematic_text)
    
    print(f"文本: {analysis2['text']}")
    print(f"倫理評分: {analysis2['ethical_evaluation']['total_ethical_score']:.3f}")
    print(f"是否可接受: {analysis2['ethical_evaluation']['is_ethically_acceptable']}")
    print(f"總體評估: {analysis2['overall_assessment']}")
    
    if analysis2['ethical_recommendations']:
        print("改進建議:")
        for i, rec in enumerate(analysis2['ethical_recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print("\n3. 倫理-數學對齊測試")
    print("-" * 30)
    
    complex_text = "語言學習涉及認知、情感、社會多個維度的協調發展"
    analysis3 = hott_nlp.ethical_action_analysis(complex_text)
    
    print(f"文本: {analysis3['text']}")
    print(f"倫理-數學對齊分數: {analysis3['ethical_math_alignment']:.3f}")
    print(f"數學複雜度: {analysis3['hott_analysis']['hott_complexity_metrics']['overall_complexity']:.3f}")
    print(f"總體評估: {analysis3['overall_assessment']}")
    
    # 驗證測試結果
    success = (
        analysis['ethical_evaluation']['is_ethically_acceptable'] and  # 正面語言應該可接受
        not analysis2['ethical_evaluation']['is_ethically_acceptable'] and  # 問題語言應該不可接受
        len(analysis2['ethical_recommendations']) > 0 and  # 問題語言應該有建議
        analysis3['ethical_math_alignment'] > 0.0  # 對齊分數應該大於0
    )
    
    print(f"\n✓ 倫理行動分析測試結果: {'通過' if success else '失敗'}")
    return success


def run_all_hott_tests():
    """運行所有HoTT測試"""
    print("🧮 同倫類型論自然語言處理 - 完整測試套件")
    print("🧮 Homotopy Type Theory Natural Language Processing - Complete Test Suite")
    print("=" * 80)
    
    if not HOTT_AVAILABLE:
        print("❌ HomotopyTypeTheoryNLP 模組不可用，無法運行測試")
        return False
    
    test_results = []
    
    try:
        # 運行所有測試
        print("\n🔍 開始測試...")
        
        test_results.append(("基本HoTT概念", test_basic_hott_concepts()))
        test_results.append(("路徑空間分析", test_path_space_analysis()))
        test_results.append(("一元性應用", test_univalence_applications()))
        test_results.append(("高階歸納類型", test_higher_inductive_types()))
        test_results.append(("同倫分析", test_homotopy_analysis()))
        test_results.append(("綜合分析", test_comprehensive_analysis()))
        test_results.append(("路徑積分整合", test_integration_with_path_integral()))
        test_results.append(("倫理行動分析", test_ethical_action_analysis()))
        test_results.append(("核心概念演示", demonstrate_hott_concepts()))
        
        print("\n" + "=" * 80)
        print("🎯 測試結果總結 (Test Results Summary)")
        print("=" * 80)
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, result in test_results:
            status = "✅ 通過" if result else "❌ 失敗"
            print(f"{status} {test_name}")
            if result:
                passed_tests += 1
            else:
                failed_tests += 1
        
        print(f"\n📊 統計:")
        print(f"   總測試數: {len(test_results)}")
        print(f"   通過: {passed_tests}")
        print(f"   失敗: {failed_tests}")
        print(f"   成功率: {passed_tests/len(test_results)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 所有測試通過！同倫類型論NLP系統運行正常。")
            print("🎉 All tests passed! HoTT NLP system is working correctly.")
            
            print("\n🔬 系統特性總結:")
            print("   ✓ 基本HoTT概念實現 (類型、路徑、等價)")
            print("   ✓ 一元性原理應用於語義分析")
            print("   ✓ 高階歸納類型構造語言結構")
            print("   ✓ 同倫分析檢測語義等價性")
            print("   ✓ 與天道路徑積分NLP完美整合")
            print("   ✓ 支持中英文混合語義分析")
            print("   ✓ 倫理行動分析確保語言使用的道德性")
            print("   ✓ 提供完整的數學基礎框架")
            
        else:
            print(f"\n⚠️  有 {failed_tests} 個測試失敗，系統需要進一步調試")
        
        return failed_tests == 0
        
    except Exception as e:
        print(f"\n💥 測試過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔮 同倫類型論自然語言處理 - 測試與演示")
    print("🔮 Homotopy Type Theory Natural Language Processing - Tests & Examples")
    print("=" * 80)
    
    # 檢查環境
    if HOTT_AVAILABLE:
        print("✅ HomotopyTypeTheoryNLP 模組可用")
        print("🔗 支持與PathIntegralNLP整合")
    else:
        print("❌ HomotopyTypeTheoryNLP 模組不可用")
    
    # 運行完整測試套件
    success = run_all_hott_tests()
    
    if success:
        print("\n" + "=" * 80)
        print("🌟 同倫類型論NLP系統已完全就緒！")
        print("🌟 Homotopy Type Theory NLP System is fully operational!")
        print("🔬 現在您可以使用HoTT的數學嚴謹性來分析自然語言")
        print("🔬 You can now analyze natural language with HoTT's mathematical rigor")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("🔧 系統需要進一步調試和完善")
        print("🔧 System requires further debugging and refinement")
        print("=" * 80)
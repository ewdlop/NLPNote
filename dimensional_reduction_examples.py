#!/usr/bin/env python3
"""
降維打擊和Flattening Earth 實例演示
Dimensional Reduction Attack and Flattening Earth Examples

這個文件提供了使用降維攻擊和地球扁平化功能的實際例子。
This file provides practical examples of using dimensional reduction attack and earth flattening functionality.

Usage:
    python dimensional_reduction_examples.py
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from DimensionalReductionAttack import (
        DimensionalAttackOrchestrator, 
        EarthFlattener,
        generate_sample_high_dimensional_data
    )
    MAIN_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import main module: {e}")
    MAIN_MODULE_AVAILABLE = False


def example_1_word_embedding_attack():
    """示例1: 詞向量降維攻擊"""
    print("\n" + "="*60)
    print("🎯 示例1: 詞向量降維攻擊 (Word Embedding Dimensional Attack)")
    print("="*60)
    
    if not MAIN_MODULE_AVAILABLE:
        print("Main module not available, skipping example")
        return
    
    # 模擬300維的詞向量數據
    print("1. 生成模擬詞向量數據...")
    vocabulary_size = 1000
    embedding_dim = 300
    word_embeddings = generate_sample_high_dimensional_data(vocabulary_size, embedding_dim)
    
    print(f"   詞彙表大小: {vocabulary_size}")
    print(f"   原始向量維度: {embedding_dim}")
    
    # 創建攻擊協調器
    orchestrator = DimensionalAttackOrchestrator()
    
    # 執行不同的降維攻擊
    print("\n2. 執行降維攻擊...")
    
    # PCA攻擊 - 降維到50維用於特徵提取
    if 'pca' in orchestrator.list_available_attacks():
        pca_result = orchestrator.execute_dimensional_attack(
            data=word_embeddings,
            attack_method='pca',
            target_dimensions=50
        )
        print(f"   PCA攻擊完成: 保留 {pca_result.data_preserved_ratio:.1%} 的信息")
    
    # t-SNE攻擊 - 降維到2維用於可視化
    if 'tsne' in orchestrator.list_available_attacks():
        tsne_result = orchestrator.execute_dimensional_attack(
            data=word_embeddings,
            attack_method='tsne', 
            target_dimensions=2
        )
        print(f"   t-SNE攻擊完成: 生成2維可視化表示")
    
    print("\n✅ 詞向量降維攻擊示例完成!")


def example_2_document_feature_attack():
    """示例2: 文檔特徵降維攻擊"""
    print("\n" + "="*60)
    print("📚 示例2: 文檔特徵降維攻擊 (Document Feature Dimensional Attack)")
    print("="*60)
    
    if not MAIN_MODULE_AVAILABLE:
        print("Main module not available, skipping example")
        return
    
    # 模擬TF-IDF特徵矩陣
    print("1. 生成模擬TF-IDF特徵...")
    n_documents = 500
    vocabulary_size = 10000
    tfidf_features = np.random.rand(n_documents, vocabulary_size) * 0.1  # 稀疏特徵
    
    print(f"   文檔數量: {n_documents}")
    print(f"   詞彙表大小: {vocabulary_size}")
    print(f"   特徵矩陣形狀: {tfidf_features.shape}")
    
    # 執行攻擊
    orchestrator = DimensionalAttackOrchestrator()
    
    print("\n2. 執行特徵降維攻擊...")
    
    # 使用PCA降維到更合理的維度
    if 'pca' in orchestrator.list_available_attacks():
        feature_result = orchestrator.execute_dimensional_attack(
            data=tfidf_features,
            attack_method='pca',
            target_dimensions=100
        )
        
        print(f"   原始特徵: {feature_result.original_dimensions}維")
        print(f"   攻擊後特徵: {feature_result.reduced_dimensions}維")
        print(f"   壓縮比例: {feature_result.attack_effectiveness:.1%}")
        print(f"   信息保留: {feature_result.data_preserved_ratio:.1%}")
    
    print("\n✅ 文檔特徵降維攻擊示例完成!")


def example_3_earth_flattening():
    """示例3: 地球扁平化演示"""
    print("\n" + "="*60)
    print("🌍 示例3: 地球扁平化演示 (Earth Flattening Demonstration)")
    print("="*60)
    
    if not MAIN_MODULE_AVAILABLE:
        print("Main module not available, skipping example")
        return
    
    flattener = EarthFlattener()
    
    # 3.1 複雜配置文件扁平化
    print("\n3.1 複雜配置文件扁平化...")
    
    complex_config = {
        'application': {
            'name': 'NLP_System',
            'version': '1.0.0',
            'database': {
                'primary': {
                    'host': 'localhost',
                    'port': 5432,
                    'credentials': {
                        'username': 'nlp_user',
                        'password': 'secure_pass',
                        'auth_method': 'password'
                    }
                },
                'backup': {
                    'host': 'backup.example.com',
                    'port': 5433
                }
            },
            'api': {
                'endpoints': {
                    'v1': {
                        'users': '/api/v1/users',
                        'documents': '/api/v1/documents',
                        'analysis': '/api/v1/analysis'
                    },
                    'v2': {
                        'users': '/api/v2/users',
                        'ml_models': '/api/v2/models'
                    }
                },
                'rate_limiting': {
                    'requests_per_minute': 1000,
                    'burst_limit': 1500
                }
            }
        }
    }
    
    config_result = flattener.flatten_nested_dict(complex_config, separator='.')
    
    print(f"   原始配置層級深度: {config_result.metadata['original_depth']}")
    print(f"   扁平化後鍵數量: {config_result.metadata['flattened_keys']}")
    print(f"   複雜度減少: {config_result.complexity_reduction:.1%}")
    
    print("\n   部分扁平化結果:")
    for i, (key, value) in enumerate(list(config_result.flattened_structure.items())[:5]):
        print(f"   {key}: {value}")
    print("   ...")
    
    # 3.2 嵌套數據結構扁平化
    print("\n3.2 嵌套數據結構扁平化...")
    
    nested_data = [
        [1, 2, [3, 4]],
        [5, [6, [7, 8, 9]]],
        [[10, 11], 12],
        [13, [14, [15, [16]]]]
    ]
    
    list_result = flattener.flatten_nested_list(nested_data)
    
    print(f"   原始列表: {nested_data}")
    print(f"   扁平化後: {list_result.flattened_structure}")
    print(f"   原始深度: {list_result.metadata['original_depth']}")
    print(f"   複雜度減少: {list_result.complexity_reduction:.1%}")
    
    # 3.3 複雜語言結構扁平化
    print("\n3.3 複雜語言結構扁平化...")
    
    complex_text = """
    Natural language processing, which encompasses a wide variety of computational linguistics techniques, 
    machine learning algorithms, and artificial intelligence methodologies, has become increasingly important 
    in the modern era of big data and digital transformation, particularly in applications such as sentiment 
    analysis, machine translation, information extraction, and question-answering systems that require 
    sophisticated understanding of human language nuances, contextual meanings, and semantic relationships.
    """
    
    text_result = flattener.flatten_linguistic_structure(complex_text.strip(), max_sentence_length=15)
    
    print(f"   原始文本長度: {len(complex_text.strip())} 字符")
    print(f"   原始句子數: {text_result.metadata['original_sentences']}")
    print(f"   扁平化後句子數: {text_result.metadata['flattened_sentences']}")
    print(f"   複雜度減少: {text_result.complexity_reduction:.1%}")
    print(f"   信息損失估計: {text_result.information_loss:.1%}")
    
    print(f"\n   扁平化後文本:\n   {text_result.flattened_structure}")
    
    print("\n✅ 地球扁平化演示完成!")


def example_4_combined_attack():
    """示例4: 組合攻擊演示"""
    print("\n" + "="*60)
    print("⚡ 示例4: 組合攻擊演示 (Combined Attack Demonstration)")
    print("="*60)
    
    if not MAIN_MODULE_AVAILABLE:
        print("Main module not available, skipping example")
        return
    
    # 創建多層次的複雜數據結構
    print("1. 創建複雜數據結構...")
    
    # 高維數據
    high_dim_data = generate_sample_high_dimensional_data(n_samples=200, n_dimensions=30)
    
    # 複雜嵌套結構
    nested_structure = {
        'experiment_data': {
            'features': high_dim_data.tolist(),  # 轉換為列表便於序列化
            'metadata': {
                'dimensions': high_dim_data.shape[1],
                'samples': high_dim_data.shape[0],
                'creation_time': '2024-12-22',
                'parameters': {
                    'algorithm': 'multivariate_normal',
                    'clusters': 3,
                    'noise_level': 0.1
                }
            }
        }
    }
    
    print(f"   數據維度: {high_dim_data.shape}")
    print(f"   嵌套結構層級: 深層嵌套")
    
    # 執行組合攻擊
    print("\n2. 執行組合攻擊...")
    
    orchestrator = DimensionalAttackOrchestrator()
    flattener = EarthFlattener()
    
    # 第一階段: 降維攻擊
    dimensional_results = orchestrator.execute_combined_attack(high_dim_data)
    
    # 第二階段: 結構扁平化  
    structure_result = flattener.flatten_nested_dict(nested_structure)
    
    print(f"\n3. 攻擊結果摘要:")
    print(f"   維度攻擊方法數: {len(dimensional_results)}")
    print(f"   結構複雜度減少: {structure_result.complexity_reduction:.1%}")
    
    # 計算總體攻擊效果
    if dimensional_results:
        avg_attack_effectiveness = np.mean([
            result.attack_effectiveness for result in dimensional_results.values()
        ])
        avg_data_preservation = np.mean([
            result.data_preserved_ratio for result in dimensional_results.values()
        ])
        
        print(f"   平均攻擊效果: {avg_attack_effectiveness:.1%}")
        print(f"   平均數據保留: {avg_data_preservation:.1%}")
    
    print("\n✅ 組合攻擊演示完成!")


def example_5_nlp_pipeline_integration():
    """示例5: NLP流水線整合"""
    print("\n" + "="*60)
    print("🔗 示例5: NLP流水線整合 (NLP Pipeline Integration)")
    print("="*60)
    
    if not MAIN_MODULE_AVAILABLE:
        print("Main module not available, skipping example")
        return
    
    # 模擬NLP流水線數據
    print("1. 模擬NLP流水線數據...")
    
    # 文本數據
    sample_texts = [
        "Natural language processing is fascinating.",
        "Machine learning algorithms can reduce dimensionality effectively.",
        "The concept of dimensional reduction attack is inspired by science fiction.",
        "Flattening complex data structures improves computational efficiency.",
        "Text analysis benefits from dimensional reduction techniques."
    ]
    
    # 模擬從文本提取的高維特徵 (例如BERT embeddings)
    text_features = generate_sample_high_dimensional_data(len(sample_texts), 768)  # BERT維度
    
    print(f"   文本數量: {len(sample_texts)}")
    print(f"   特徵維度: {text_features.shape[1]} (模擬BERT)")
    
    # 模擬複雜的NLP處理結果
    nlp_results = {
        'text_analysis': {
            'sentiment': {
                'positive': 0.8,
                'negative': 0.1,
                'neutral': 0.1
            },
            'entities': {
                'person': ['researchers', 'scientists'],
                'technology': ['NLP', 'machine learning', 'BERT'],
                'concepts': ['dimensionality', 'reduction', 'flattening']
            },
            'embeddings': {
                'sentence_embeddings': text_features.tolist(),
                'metadata': {
                    'model': 'BERT',
                    'dimensions': 768,
                    'processing_time': '2.5s'
                }
            }
        }
    }
    
    print("\n2. 執行整合降維攻擊...")
    
    orchestrator = DimensionalAttackOrchestrator()
    flattener = EarthFlattener()
    
    # 對文本特徵執行降維
    if 'pca' in orchestrator.list_available_attacks():
        # Adjust target dimensions based on available data
        max_dims = min(text_features.shape[0], text_features.shape[1]) - 1
        target_dims = min(50, max_dims)  # 確保不超過最大可用維度
        
        feature_attack = orchestrator.execute_dimensional_attack(
            data=text_features,
            attack_method='pca',
            target_dimensions=target_dims  # 動態調整目標維度
        )
        
        print(f"   文本特徵降維: {feature_attack.original_dimensions}D → {feature_attack.reduced_dimensions}D")
        print(f"   信息保留: {feature_attack.data_preserved_ratio:.1%}")
    
    # 扁平化NLP結果結構
    structure_attack = flattener.flatten_nested_dict(nlp_results)
    
    print(f"   結構扁平化: {structure_attack.metadata['original_depth']} 層 → 1 層")
    print(f"   複雜度減少: {structure_attack.complexity_reduction:.1%}")
    
    print("\n3. 整合後的流水線效益:")
    print("   ✓ 降低存儲需求")
    print("   ✓ 提高計算效率") 
    print("   ✓ 簡化數據訪問")
    print("   ✓ 便於後續處理")
    
    print("\n✅ NLP流水線整合示例完成!")


def main():
    """主函數 - 運行所有示例"""
    print("🌟 降維打擊和Flattening Earth 實例演示")
    print("🌟 Dimensional Reduction Attack and Flattening Earth Examples")
    print("="*80)
    
    if not MAIN_MODULE_AVAILABLE:
        print("❌ 錯誤: 無法導入主模組，請確保 DimensionalReductionAttack.py 在當前目錄")
        print("❌ Error: Cannot import main module, ensure DimensionalReductionAttack.py is in current directory")
        return
    
    try:
        # 運行所有示例
        example_1_word_embedding_attack()
        example_2_document_feature_attack()
        example_3_earth_flattening()
        example_4_combined_attack()
        example_5_nlp_pipeline_integration()
        
        print("\n" + "="*80)
        print("🎉 所有示例演示完成! (All examples completed!)")
        print("🎉 降維攻擊和地球扁平化功能展示結束")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 運行示例時發生錯誤: {e}")
        print("請檢查依賴項是否正確安裝")


if __name__ == "__main__":
    main()
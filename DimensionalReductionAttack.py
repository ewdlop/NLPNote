"""
降維打擊和Flattening Earth - Dimensional Reduction Attack and Flattening Earth
==============================================================================

This module implements dimensional reduction techniques inspired by the concept of 
"dimensional reduction attack" (降維打擊) from Liu Cixin's Three-Body Problem series,
applied to Natural Language Processing and data flattening operations.

In the context of NLP:
- "Dimensional Reduction Attack" refers to reducing high-dimensional linguistic data 
  (word embeddings, feature vectors) to lower dimensions while preserving essential information
- "Flattening Earth" refers to simplifying complex nested linguistic structures 
  into flat, manageable representations

Author: NLP Note Project
Date: 2024-12-22
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None


@dataclass
class DimensionalAttackResult:
    """降維攻擊結果 (Dimensional Attack Result)"""
    original_dimensions: int
    reduced_dimensions: int
    attack_method: str
    data_preserved_ratio: float  # 0.0 to 1.0
    attack_effectiveness: float  # 0.0 to 1.0 (higher = more effective reduction)
    reduced_data: np.ndarray
    metadata: Dict[str, Any] = None


@dataclass 
class FlatteningResult:
    """扁平化結果 (Flattening Result)"""
    original_structure: Any
    flattened_structure: Any
    flattening_method: str
    complexity_reduction: float  # 0.0 to 1.0
    information_loss: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None


class DimensionalReductionAttacker(ABC):
    """降維攻擊器抽象基類 (Abstract Dimensional Reduction Attacker)"""
    
    @abstractmethod
    def attack(self, high_dimensional_data: np.ndarray, target_dimensions: int) -> DimensionalAttackResult:
        """執行降維攻擊 (Execute dimensional reduction attack)"""
        pass
    
    @abstractmethod
    def get_attack_name(self) -> str:
        """獲取攻擊名稱 (Get attack name)"""
        pass


class PCAAttacker(DimensionalReductionAttacker):
    """PCA降維攻擊器 (PCA Dimensional Reduction Attacker)"""
    
    def __init__(self, preserve_variance: float = 0.95):
        self.preserve_variance = preserve_variance
        self.pca_model = None
        
    def attack(self, high_dimensional_data: np.ndarray, target_dimensions: int = None) -> DimensionalAttackResult:
        """使用PCA執行降維攻擊"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for PCA attack. Please install: pip install scikit-learn")
        
        original_dims = high_dimensional_data.shape[1]
        
        if target_dimensions is None:
            # Auto-determine dimensions based on variance preservation
            pca_full = PCA()
            pca_full.fit(high_dimensional_data)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            target_dimensions = np.argmax(cumsum_variance >= self.preserve_variance) + 1
        
        # Execute the dimensional attack
        self.pca_model = PCA(n_components=target_dimensions)
        reduced_data = self.pca_model.fit_transform(high_dimensional_data)
        
        # Calculate attack effectiveness
        explained_variance = np.sum(self.pca_model.explained_variance_ratio_)
        attack_effectiveness = (original_dims - target_dimensions) / original_dims
        
        return DimensionalAttackResult(
            original_dimensions=original_dims,
            reduced_dimensions=target_dimensions,
            attack_method="PCA_Attack",
            data_preserved_ratio=explained_variance,
            attack_effectiveness=attack_effectiveness,
            reduced_data=reduced_data,
            metadata={
                "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
                "cumulative_variance": explained_variance
            }
        )
    
    def get_attack_name(self) -> str:
        return "PCA_Dimensional_Attack"


class TSNEAttacker(DimensionalReductionAttacker):
    """t-SNE降維攻擊器 (t-SNE Dimensional Reduction Attacker)"""
    
    def __init__(self, perplexity: float = 30.0, learning_rate: str = 'auto'):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.tsne_model = None
        
    def attack(self, high_dimensional_data: np.ndarray, target_dimensions: int = 2) -> DimensionalAttackResult:
        """使用t-SNE執行降維攻擊"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for t-SNE attack. Please install: pip install scikit-learn")
        
        original_dims = high_dimensional_data.shape[1]
        
        # Execute the dimensional attack
        self.tsne_model = TSNE(
            n_components=target_dimensions,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        reduced_data = self.tsne_model.fit_transform(high_dimensional_data)
        
        # t-SNE doesn't provide explained variance, so we estimate effectiveness
        attack_effectiveness = (original_dims - target_dimensions) / original_dims
        
        return DimensionalAttackResult(
            original_dimensions=original_dims,
            reduced_dimensions=target_dimensions,
            attack_method="tSNE_Attack",
            data_preserved_ratio=0.8,  # Estimated for t-SNE
            attack_effectiveness=attack_effectiveness,
            reduced_data=reduced_data,
            metadata={
                "perplexity": self.perplexity,
                "learning_rate": self.learning_rate
            }
        )
    
    def get_attack_name(self) -> str:
        return "tSNE_Dimensional_Attack"


class UMAPAttacker(DimensionalReductionAttacker):
    """UMAP降維攻擊器 (UMAP Dimensional Reduction Attacker)"""
    
    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None
        
    def attack(self, high_dimensional_data: np.ndarray, target_dimensions: int = 2) -> DimensionalAttackResult:
        """使用UMAP執行降維攻擊"""
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn is required for UMAP attack. Please install: pip install umap-learn")
        
        original_dims = high_dimensional_data.shape[1]
        
        # Execute the dimensional attack
        self.umap_model = umap.UMAP(
            n_components=target_dimensions,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42
        )
        
        reduced_data = self.umap_model.fit_transform(high_dimensional_data)
        
        attack_effectiveness = (original_dims - target_dimensions) / original_dims
        
        return DimensionalAttackResult(
            original_dimensions=original_dims,
            reduced_dimensions=target_dimensions,
            attack_method="UMAP_Attack",
            data_preserved_ratio=0.85,  # Estimated for UMAP
            attack_effectiveness=attack_effectiveness,
            reduced_data=reduced_data,
            metadata={
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist
            }
        )
    
    def get_attack_name(self) -> str:
        return "UMAP_Dimensional_Attack"


class EarthFlattener:
    """地球扁平化器 (Earth Flattener) - For flattening complex data structures"""
    
    @staticmethod
    def flatten_nested_dict(nested_dict: Dict[str, Any], separator: str = "_") -> FlatteningResult:
        """扁平化嵌套字典 (Flatten nested dictionary)"""
        
        def _flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = _flatten_dict(nested_dict, sep=separator)
        
        # Calculate complexity reduction
        original_depth = EarthFlattener._calculate_dict_depth(nested_dict)
        complexity_reduction = max(0, (original_depth - 1) / original_depth) if original_depth > 1 else 0
        
        # Estimate information loss (minimal for dictionary flattening)
        information_loss = 0.1 if original_depth > 2 else 0.0
        
        return FlatteningResult(
            original_structure=nested_dict,
            flattened_structure=flattened,
            flattening_method="nested_dict_flattening",
            complexity_reduction=complexity_reduction,
            information_loss=information_loss,
            metadata={
                "original_depth": original_depth,
                "original_keys": len(nested_dict),
                "flattened_keys": len(flattened),
                "separator": separator
            }
        )
    
    @staticmethod
    def flatten_nested_list(nested_list: List[Any]) -> FlatteningResult:
        """扁平化嵌套列表 (Flatten nested list)"""
        
        def _flatten_list(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(_flatten_list(item))
                else:
                    result.append(item)
            return result
        
        flattened = _flatten_list(nested_list)
        
        # Calculate complexity reduction
        original_depth = EarthFlattener._calculate_list_depth(nested_list)
        complexity_reduction = max(0, (original_depth - 1) / original_depth) if original_depth > 1 else 0
        
        # Minimal information loss for list flattening
        information_loss = 0.05 if original_depth > 2 else 0.0
        
        return FlatteningResult(
            original_structure=nested_list,
            flattened_structure=flattened,
            flattening_method="nested_list_flattening",
            complexity_reduction=complexity_reduction,
            information_loss=information_loss,
            metadata={
                "original_depth": original_depth,
                "original_length": len(nested_list),
                "flattened_length": len(flattened)
            }
        )
    
    @staticmethod
    def flatten_linguistic_structure(text: str, max_sentence_length: int = 50) -> FlatteningResult:
        """扁平化語言結構 (Flatten linguistic structure)"""
        
        # Simple sentence splitting and flattening
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Flatten long sentences
        flattened_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > max_sentence_length:
                # Break long sentences at commas or semicolons
                parts = sentence.replace(',', '.').replace(';', '.').split('.')
                flattened_sentences.extend([p.strip() for p in parts if p.strip()])
            else:
                flattened_sentences.append(sentence)
        
        flattened_text = '. '.join(flattened_sentences) + '.'
        
        # Calculate metrics
        original_complexity = np.mean([len(s.split()) for s in sentences])
        flattened_complexity = np.mean([len(s.split()) for s in flattened_sentences])
        complexity_reduction = max(0, (original_complexity - flattened_complexity) / original_complexity)
        
        # Estimate information loss
        information_loss = min(0.3, complexity_reduction * 0.5)
        
        return FlatteningResult(
            original_structure=text,
            flattened_structure=flattened_text,
            flattening_method="linguistic_structure_flattening",
            complexity_reduction=complexity_reduction,
            information_loss=information_loss,
            metadata={
                "original_sentences": len(sentences),
                "flattened_sentences": len(flattened_sentences),
                "original_avg_length": original_complexity,
                "flattened_avg_length": flattened_complexity,
                "max_sentence_length": max_sentence_length
            }
        )
    
    @staticmethod
    def _calculate_dict_depth(d: Dict) -> int:
        """計算字典深度"""
        if not isinstance(d, dict) or not d:
            return 0
        return 1 + max(EarthFlattener._calculate_dict_depth(v) if isinstance(v, dict) else 0 for v in d.values())
    
    @staticmethod
    def _calculate_list_depth(lst: List) -> int:
        """計算列表深度"""
        if not isinstance(lst, list) or not lst:
            return 0
        return 1 + max(EarthFlattener._calculate_list_depth(item) if isinstance(item, list) else 0 for item in lst)


class DimensionalAttackOrchestrator:
    """降維攻擊協調器 (Dimensional Attack Orchestrator)"""
    
    def __init__(self):
        self.attackers = {}
        self.flattener = EarthFlattener()
        
        # Register available attackers
        if SKLEARN_AVAILABLE:
            self.attackers['pca'] = PCAAttacker()
            self.attackers['tsne'] = TSNEAttacker()
        
        if UMAP_AVAILABLE:
            self.attackers['umap'] = UMAPAttacker()
    
    def list_available_attacks(self) -> List[str]:
        """列出可用的攻擊方法"""
        return list(self.attackers.keys())
    
    def execute_dimensional_attack(self, 
                                 data: np.ndarray, 
                                 attack_method: str = 'pca',
                                 target_dimensions: int = 2) -> DimensionalAttackResult:
        """執行指定的降維攻擊"""
        if attack_method not in self.attackers:
            available = self.list_available_attacks()
            raise ValueError(f"Attack method '{attack_method}' not available. Available methods: {available}")
        
        attacker = self.attackers[attack_method]
        result = attacker.attack(data, target_dimensions)
        
        print(f"🎯 Dimensional Attack Executed!")
        print(f"   Method: {result.attack_method}")
        print(f"   Dimensions: {result.original_dimensions} → {result.reduced_dimensions}")
        print(f"   Attack Effectiveness: {result.attack_effectiveness:.2%}")
        print(f"   Data Preserved: {result.data_preserved_ratio:.2%}")
        
        return result
    
    def execute_combined_attack(self, data: np.ndarray, flatten_first: bool = True) -> Dict[str, Any]:
        """執行組合攻擊 - 結合降維和扁平化"""
        results = {}
        
        # Execute all available dimensional attacks
        for method in self.list_available_attacks():
            try:
                results[method] = self.execute_dimensional_attack(data, method)
            except Exception as e:
                print(f"Attack {method} failed: {e}")
        
        print(f"\n🌍 Combined Dimensional Attack Completed!")
        print(f"   Executed {len(results)} attacks successfully")
        
        return results
    
    def visualize_attack_results(self, results: Dict[str, DimensionalAttackResult], 
                               original_data: np.ndarray = None):
        """可視化攻擊結果"""
        if not results:
            print("No results to visualize")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            n_methods = len(results)
            fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
            
            if n_methods == 1:
                axes = [axes]
            
            for i, (method, result) in enumerate(results.items()):
                ax = axes[i]
                
                if result.reduced_dimensions == 2:
                    scatter = ax.scatter(result.reduced_data[:, 0], result.reduced_data[:, 1], 
                                       alpha=0.6, s=50)
                    ax.set_title(f'{method.upper()}\nEffectiveness: {result.attack_effectiveness:.2%}')
                    ax.set_xlabel('Dimension 1')
                    ax.set_ylabel('Dimension 2')
                    ax.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.suptitle('🎯 Dimensional Reduction Attack Results 🌍', fontsize=16, y=1.02)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def generate_sample_high_dimensional_data(n_samples: int = 500, n_dimensions: int = 50) -> np.ndarray:
    """生成樣本高維數據用於測試"""
    np.random.seed(42)
    
    # Create structured high-dimensional data with some patterns
    cluster1 = np.random.multivariate_normal([0]*n_dimensions, np.eye(n_dimensions), n_samples//3)
    cluster2 = np.random.multivariate_normal([3]*n_dimensions, np.eye(n_dimensions), n_samples//3)
    cluster3 = np.random.multivariate_normal([-2]*n_dimensions, np.eye(n_dimensions), n_samples//3)
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Add some noise and patterns
    data += np.random.normal(0, 0.1, data.shape)
    
    return data


def demonstrate_dimensional_attack():
    """演示降維攻擊功能"""
    print("=" * 60)
    print("🎯 DIMENSIONAL REDUCTION ATTACK DEMONSTRATION 🌍")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating high-dimensional data...")
    high_dim_data = generate_sample_high_dimensional_data(n_samples=300, n_dimensions=25)
    print(f"   Generated data shape: {high_dim_data.shape}")
    
    # Create orchestrator
    orchestrator = DimensionalAttackOrchestrator()
    available_attacks = orchestrator.list_available_attacks()
    print(f"   Available attacks: {available_attacks}")
    
    # Execute attacks
    print("\n2. Executing dimensional attacks...")
    results = orchestrator.execute_combined_attack(high_dim_data)
    
    # Demonstrate flattening
    print("\n3. Demonstrating Earth Flattening...")
    flattener = EarthFlattener()
    
    # Flatten nested dictionary
    nested_dict = {
        'level1': {
            'level2a': {
                'level3': {'data': [1, 2, 3]},
                'other': 'value'
            },
            'level2b': {'simple': 42}
        },
        'top_level': 'direct_value'
    }
    
    dict_result = flattener.flatten_nested_dict(nested_dict)
    print(f"   🗂️  Dict flattening complexity reduction: {dict_result.complexity_reduction:.2%}")
    
    # Flatten linguistic structure  
    complex_text = ("This is a very long and complex sentence with multiple clauses, subclauses, and embedded thoughts that make it difficult to understand. "
                   "It contains various ideas, concepts, and elaborations that could be simplified. "
                   "Such complexity often hinders comprehension and should be reduced for better clarity.")
    
    text_result = flattener.flatten_linguistic_structure(complex_text, max_sentence_length=10)
    print(f"   📝 Text flattening complexity reduction: {text_result.complexity_reduction:.2%}")
    
    print("\n4. Attack Summary:")
    for method, result in results.items():
        print(f"   {method.upper()}: {result.original_dimensions}D → {result.reduced_dimensions}D "
              f"(Effectiveness: {result.attack_effectiveness:.2%})")
    
    print("\n🎉 Dimensional Attack Demonstration Complete!")
    print("=" * 60)
    
    return results, dict_result, text_result


if __name__ == "__main__":
    # Run demonstration
    demonstrate_dimensional_attack()
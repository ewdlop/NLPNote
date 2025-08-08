#!/usr/bin/env python3
"""
一本正經地「胡說八道」示例 (Serious Nonsense Examples)

This script demonstrates various uses of the SeriousNonsenseGenerator
to create academic-sounding but meaningless content.
"""

from SeriousNonsenseGenerator import (
    SeriousNonsenseGenerator, 
    GenerationContext, 
    AcademicStyle, 
    Language
)

def demo_basic_usage():
    """Demonstrate basic usage of the generator"""
    print("=== Basic Usage Demo ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    # Simple English generation
    result = generator.generate_nonsense()
    print("Default English nonsense:")
    print(result)
    print()
    
    # Simple Chinese generation
    zh_context = GenerationContext(language=Language.CHINESE)
    zh_result = generator.generate_nonsense(zh_context)
    print("Default Chinese nonsense:")
    print(zh_result)
    print()

def demo_different_styles():
    """Demonstrate different academic styles"""
    print("=== Different Academic Styles ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    styles = [
        (AcademicStyle.SCIENTIFIC, "科學風格"),
        (AcademicStyle.PHILOSOPHICAL, "哲學風格"),
        (AcademicStyle.TECHNICAL, "技術風格"),
        (AcademicStyle.THEORETICAL, "理論風格"),
        (AcademicStyle.LINGUISTIC, "語言學風格")
    ]
    
    for style, name in styles:
        print(f"【{name} / {style.value.upper()}】")
        
        # English
        en_context = GenerationContext(
            style=style, 
            language=Language.ENGLISH,
            length="short"
        )
        en_title = generator.generate_academic_title(en_context)
        en_content = generator.generate_nonsense(en_context)
        
        print(f"EN Title: {en_title}")
        print(f"EN: {en_content[:150]}...")
        
        # Chinese
        zh_context = GenerationContext(
            style=style, 
            language=Language.CHINESE,
            length="short"
        )
        zh_title = generator.generate_academic_title(zh_context)
        zh_content = generator.generate_nonsense(zh_context)
        
        print(f"中文標題：{zh_title}")
        print(f"中文：{zh_content[:100]}...")
        print()

def demo_complexity_levels():
    """Demonstrate different complexity levels"""
    print("=== Complexity Levels ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    complexities = [
        (0.3, "Simple", "簡單"),
        (0.5, "Medium", "中等"),
        (0.8, "Complex", "複雜")
    ]
    
    for complexity, en_name, zh_name in complexities:
        print(f"【{en_name} ({zh_name}) - Complexity: {complexity}】")
        
        context = GenerationContext(
            complexity=complexity,
            language=Language.ENGLISH,
            length="short"
        )
        
        result = generator.generate_nonsense(context)
        print(f"English: {result}")
        
        zh_context = GenerationContext(
            complexity=complexity,
            language=Language.CHINESE,
            length="short"
        )
        
        zh_result = generator.generate_nonsense(zh_context)
        print(f"中文：{zh_result}")
        print()

def demo_title_generation():
    """Demonstrate academic title generation"""
    print("=== Academic Title Generation ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    print("English Academic Titles:")
    for i in range(5):
        title = generator.generate_academic_title()
        print(f"{i+1}. {title}")
    
    print("\n中文學術標題：")
    zh_context = GenerationContext(language=Language.CHINESE)
    for i in range(5):
        title = generator.generate_academic_title(zh_context)
        print(f"{i+1}. {title}")
    print()

def demo_interactive_generation():
    """Demonstrate interactive generation"""
    print("=== Interactive Generation ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    try:
        while True:
            print("Options:")
            print("1. Generate English nonsense")
            print("2. Generate Chinese nonsense")
            print("3. Generate academic title")
            print("4. Generate complex philosophical text")
            print("5. Exit")
            
            choice = input("\nChoose an option (1-5): ").strip()
            
            if choice == "1":
                result = generator.generate_nonsense()
                print(f"\nEnglish Nonsense:\n{result}\n")
                
            elif choice == "2":
                context = GenerationContext(language=Language.CHINESE)
                result = generator.generate_nonsense(context)
                print(f"\n中文胡話：\n{result}\n")
                
            elif choice == "3":
                en_title = generator.generate_academic_title()
                zh_context = GenerationContext(language=Language.CHINESE)
                zh_title = generator.generate_academic_title(zh_context)
                print(f"\nEnglish Title: {en_title}")
                print(f"中文標題：{zh_title}\n")
                
            elif choice == "4":
                context = GenerationContext(
                    style=AcademicStyle.PHILOSOPHICAL,
                    complexity=0.9,
                    length="long"
                )
                result = generator.generate_nonsense(context)
                print(f"\nComplex Philosophical Nonsense:\n{result}\n")
                
            elif choice == "5":
                print("Goodbye! 再見！")
                break
                
            else:
                print("Invalid choice. Please choose 1-5.")
                
    except KeyboardInterrupt:
        print("\n\nInteractive session ended.")

def demo_research_paper_simulation():
    """Simulate generating a research paper abstract"""
    print("=== Research Paper Simulation ===\n")
    
    generator = SeriousNonsenseGenerator()
    
    # Generate English abstract
    en_context = GenerationContext(
        style=AcademicStyle.SCIENTIFIC,
        complexity=0.8,
        length="medium"
    )
    
    en_title = generator.generate_academic_title(en_context)
    en_abstract = generator.generate_nonsense(en_context)
    
    print("【English Research Paper】")
    print(f"Title: {en_title}")
    print(f"Abstract: {en_abstract}")
    print()
    
    # Generate Chinese abstract
    zh_context = GenerationContext(
        style=AcademicStyle.THEORETICAL,
        language=Language.CHINESE,
        complexity=0.7,
        length="medium"
    )
    
    zh_title = generator.generate_academic_title(zh_context)
    zh_abstract = generator.generate_nonsense(zh_context)
    
    print("【中文研究論文】")
    print(f"標題：{zh_title}")
    print(f"摘要：{zh_abstract}")
    print()

def main():
    """Main demonstration function"""
    print("一本正經地「胡說八道」生成器示例")
    print("Serious Nonsense Generator Examples")
    print("=" * 50)
    print()
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Different Styles", demo_different_styles),
        ("Complexity Levels", demo_complexity_levels),
        ("Title Generation", demo_title_generation),
        ("Research Paper Simulation", demo_research_paper_simulation),
    ]
    
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        demo_func()
        input("Press Enter to continue...")
    
    print("\nWould you like to try interactive generation? (y/n): ", end="")
    if input().lower().startswith('y'):
        demo_interactive_generation()
    
    print("\nDemo completed! 演示完成！")

if __name__ == "__main__":
    main()
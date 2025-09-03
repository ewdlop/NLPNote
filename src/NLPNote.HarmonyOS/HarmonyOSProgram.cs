using NLPNote.Core;

namespace NLPNote.HarmonyOS;

/// <summary>
/// HarmonyOS-specific entry point for NLP Note application.
/// Optimized for HarmonyOS/OpenHarmony platform with proper system integration.
/// </summary>
public class HarmonyOSProgram
{
    public static void Main(string[] args)
    {
        try
        {
            Console.WriteLine("=== NLP Note for HarmonyOS ===");
            Console.WriteLine(PlatformUtils.GetPlatformInfo());
            
            // Verify HarmonyOS platform detection
            if (PlatformUtils.IsHarmonyOS())
            {
                Console.WriteLine("✓ HarmonyOS platform detected successfully");
            }
            else
            {
                Console.WriteLine("⚠ HarmonyOS platform detection may not be working properly");
                Console.WriteLine("   This may be running on a compatible Linux distribution");
            }
            
            Console.WriteLine();

            // Check for HarmonyOS specific features
            CheckHarmonyOSFeatures();
            
            if (args.Length > 0)
            {
                ProcessHarmonyOSCommands(args);
            }
            else
            {
                RunInteractiveMode();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Environment.Exit(1);
        }
    }

    private static void CheckHarmonyOSFeatures()
    {
        Console.WriteLine("=== HarmonyOS Feature Detection ===");
        
        // Check for HarmonyOS environment variables
        var harmonyHome = Environment.GetEnvironmentVariable("HARMONY_HOME");
        var ohosSDK = Environment.GetEnvironmentVariable("OHOS_SDK_HOME");
        
        Console.WriteLine($"HARMONY_HOME: {harmonyHome ?? "Not set"}");
        Console.WriteLine($"OHOS_SDK_HOME: {ohosSDK ?? "Not set"}");
        Console.WriteLine($"OS Version: {Environment.OSVersion.VersionString}");
        Console.WriteLine();
    }

    private static void RunInteractiveMode()
    {
        Console.WriteLine("Welcome to NLP Note for HarmonyOS!");
        Console.WriteLine("支持鸿蒙操作系统的自然语言处理工具");
        Console.WriteLine();
        
        while (true)
        {
            Console.WriteLine("Choose an option / 选择选项:");
            Console.WriteLine("1. Text Reversal / 文本反转");
            Console.WriteLine("2. Diacritical Processing / 变音符号处理");
            Console.WriteLine("3. Chinese Text Demo / 中文文本演示");
            Console.WriteLine("4. Platform Information / 平台信息");
            Console.WriteLine("5. Exit / 退出");
            Console.Write("Enter your choice (1-5) / 输入选择 (1-5): ");

            var choice = Console.ReadLine();
            Console.WriteLine();

            switch (choice)
            {
                case "1":
                    HandleTextReversal();
                    break;
                case "2":
                    HandleDiacriticalProcessing();
                    break;
                case "3":
                    RunChineseDemo();
                    break;
                case "4":
                    ShowHarmonyOSPlatformInfo();
                    break;
                case "5":
                    Console.WriteLine("再见! / Goodbye!");
                    return;
                default:
                    Console.WriteLine("Invalid choice / 无效选择. Please try again / 请重试.");
                    break;
            }
            
            Console.WriteLine();
        }
    }

    private static void HandleTextReversal()
    {
        Console.Write("Enter text to reverse / 输入要反转的文本: ");
        var text = Console.ReadLine() ?? string.Empty;
        
        Console.Write("Enter line length (default 10) / 输入行长度 (默认10): ");
        var lineLengthStr = Console.ReadLine();
        var lineLength = int.TryParse(lineLengthStr, out var len) ? len : 10;

        var result = TextReversal.ReverseByLines(text, lineLength);
        Console.WriteLine("\nReversed text / 反转后的文本:");
        Console.WriteLine(result);
    }

    private static void HandleDiacriticalProcessing()
    {
        Console.Write("Enter text to process / 输入要处理的文本: ");
        var text = Console.ReadLine() ?? string.Empty;
        
        Console.WriteLine("Choose context / 选择上下文:");
        Console.WriteLine("1. Default / 默认");
        Console.WriteLine("2. Emotional / 情感");
        Console.WriteLine("3. Technical / 技术");
        Console.WriteLine("4. Random / 随机");
        Console.Write("Enter choice (1-4) / 输入选择 (1-4): ");
        
        var contextChoice = Console.ReadLine();
        var context = contextChoice switch
        {
            "2" => DiacriticalProcessor.ProcessingContext.Emotional,
            "3" => DiacriticalProcessor.ProcessingContext.Technical,
            "4" => DiacriticalProcessor.ProcessingContext.Random,
            _ => DiacriticalProcessor.ProcessingContext.Default
        };

        var result = DiacriticalProcessor.ApplyAscentMarks(text, context);
        Console.WriteLine($"\nProcessed text ({context}) / 处理后的文本 ({context}):");
        Console.WriteLine(result);
    }

    private static void RunChineseDemo()
    {
        Console.WriteLine("=== Chinese Text Processing Demo / 中文文本处理演示 ===");
        
        var chineseTexts = new[]
        {
            "鸿蒙操作系统是面向万物互联时代的全新分布式操作系统",
            "自然语言处理技术在人工智能领域发挥着重要作用",
            "跨平台应用开发让软件能够在不同设备上运行"
        };

        foreach (var text in chineseTexts)
        {
            Console.WriteLine($"\nOriginal / 原文: {text}");
            Console.WriteLine($"Reversed / 反转: {TextReversal.ReverseString(text)}");
            
            var processed = DiacriticalProcessor.ApplyAscentMarks(text, DiacriticalProcessor.ProcessingContext.Emotional);
            Console.WriteLine($"Processed / 处理: {processed}");
            Console.WriteLine(new string('-', 50));
        }
    }

    private static void ShowHarmonyOSPlatformInfo()
    {
        Console.WriteLine("=== HarmonyOS Platform Information / 鸿蒙平台信息 ===");
        Console.WriteLine(PlatformUtils.GetPlatformInfo());
        Console.WriteLine($"Current Platform / 当前平台: {PlatformUtils.GetCurrentPlatform()}");
        Console.WriteLine($"Is HarmonyOS / 是否为鸿蒙: {PlatformUtils.IsHarmonyOS()}");
        Console.WriteLine($"Unicode Support / Unicode支持: {PlatformUtils.SupportsUnicode()}");
        Console.WriteLine($"Line Ending / 行结束符: {PlatformUtils.GetLineEnding().Replace("\r", "\\r").Replace("\n", "\\n")}");
        Console.WriteLine($"Path Separator / 路径分隔符: {PlatformUtils.GetPathSeparator()}");
        Console.WriteLine($"Temp Directory / 临时目录: {PlatformUtils.GetTempDirectory()}");
        
        // HarmonyOS specific information
        CheckHarmonyOSFeatures();
    }

    private static void ProcessHarmonyOSCommands(string[] args)
    {
        var command = args[0].ToLowerInvariant();
        
        switch (command)
        {
            case "demo":
            case "演示":
                RunChineseDemo();
                break;
                
            case "platform":
            case "平台":
                ShowHarmonyOSPlatformInfo();
                break;
                
            case "reverse":
            case "反转":
                if (args.Length > 1)
                {
                    var lineLength = args.Length > 2 && int.TryParse(args[2], out var len) ? len : 10;
                    var result = TextReversal.ReverseByLines(args[1], lineLength);
                    Console.WriteLine($"Reversed / 反转结果: \n{result}");
                }
                else
                {
                    Console.WriteLine("Usage: reverse <text> [lineLength] / 用法: reverse <文本> [行长度]");
                }
                break;
                
            case "diacritical":
            case "变音":
                if (args.Length > 1)
                {
                    var context = args.Length > 2 && Enum.TryParse<DiacriticalProcessor.ProcessingContext>(args[2], true, out var ctx) 
                        ? ctx 
                        : DiacriticalProcessor.ProcessingContext.Default;
                    var result = DiacriticalProcessor.ApplyAscentMarks(args[1], context);
                    Console.WriteLine($"Processed / 处理结果: {result}");
                }
                else
                {
                    Console.WriteLine("Usage: diacritical <text> [context] / 用法: diacritical <文本> [上下文]");
                }
                break;
                
            default:
                Console.WriteLine($"Unknown command / 未知命令: {command}");
                Console.WriteLine("Available commands / 可用命令: demo, platform, reverse, diacritical");
                break;
        }
    }
}
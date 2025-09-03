using NLPNote.Core;

namespace NLPNote.Console;

/// <summary>
/// Cross-platform console application for NLP text processing.
/// Supports Windows, Linux, macOS, Android, and HarmonyOS.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        // Display platform information
        System.Console.WriteLine("=== NLP Note - Cross-Platform Text Processor ===");
        System.Console.WriteLine(PlatformUtils.GetPlatformInfo());
        System.Console.WriteLine();

        if (args.Length > 0)
        {
            // Command-line mode
            ProcessCommandLineArguments(args);
        }
        else
        {
            // Interactive mode
            RunInteractiveMode();
        }
    }

    private static void ProcessCommandLineArguments(string[] args)
    {
        var command = args[0].ToLowerInvariant();
        
        switch (command)
        {
            case "reverse":
                if (args.Length < 2)
                {
                    System.Console.WriteLine("Usage: nlpnote reverse <text> [lineLength]");
                    return;
                }
                
                var text = args[1];
                var lineLength = args.Length > 2 && int.TryParse(args[2], out var len) ? len : 10;
                
                var result = TextReversal.ReverseByLines(text, lineLength);
                System.Console.WriteLine("Reversed text:");
                System.Console.WriteLine(result);
                break;

            case "diacritical":
                if (args.Length < 2)
                {
                    System.Console.WriteLine("Usage: nlpnote diacritical <text> [context]");
                    return;
                }
                
                var inputText = args[1];
                var contextStr = args.Length > 2 ? args[2] : "default";
                var context = Enum.TryParse<DiacriticalProcessor.ProcessingContext>(contextStr, true, out var ctx) 
                    ? ctx 
                    : DiacriticalProcessor.ProcessingContext.Default;
                
                var processed = DiacriticalProcessor.ApplyAscentMarks(inputText, context);
                System.Console.WriteLine($"Processed text ({context}):");
                System.Console.WriteLine(processed);
                break;

            case "platform":
                System.Console.WriteLine(PlatformUtils.GetPlatformInfo());
                System.Console.WriteLine($"Current Platform: {PlatformUtils.GetCurrentPlatform()}");
                System.Console.WriteLine($"Unicode Support: {PlatformUtils.SupportsUnicode()}");
                System.Console.WriteLine($"Line Ending: {PlatformUtils.GetLineEnding().Replace("\r", "\\r").Replace("\n", "\\n")}");
                System.Console.WriteLine($"Path Separator: {PlatformUtils.GetPathSeparator()}");
                break;

            case "demo":
                RunDemo();
                break;

            default:
                ShowHelp();
                break;
        }
    }

    private static void RunInteractiveMode()
    {
        System.Console.WriteLine("Welcome to the Cross-Platform NLP Text Processor!");
        System.Console.WriteLine("Supports text processing on Windows, Linux, macOS, Android, and HarmonyOS");
        System.Console.WriteLine();
        
        while (true)
        {
            System.Console.WriteLine("Choose an option:");
            System.Console.WriteLine("1. Text Reversal");
            System.Console.WriteLine("2. Diacritical Mark Processing");
            System.Console.WriteLine("3. Platform Information");
            System.Console.WriteLine("4. Run Demo");
            System.Console.WriteLine("5. Exit");
            System.Console.Write("Enter your choice (1-5): ");

            var choice = System.Console.ReadLine();
            System.Console.WriteLine();

            switch (choice)
            {
                case "1":
                    HandleTextReversal();
                    break;
                case "2":
                    HandleDiacriticalProcessing();
                    break;
                case "3":
                    ShowPlatformInfo();
                    break;
                case "4":
                    RunDemo();
                    break;
                case "5":
                    System.Console.WriteLine("Goodbye!");
                    return;
                default:
                    System.Console.WriteLine("Invalid choice. Please try again.");
                    break;
            }
            
            System.Console.WriteLine();
        }
    }

    private static void HandleTextReversal()
    {
        System.Console.Write("Enter text to reverse: ");
        var text = System.Console.ReadLine() ?? string.Empty;
        
        System.Console.Write("Enter line length (default 10): ");
        var lineLengthStr = System.Console.ReadLine();
        var lineLength = int.TryParse(lineLengthStr, out var len) ? len : 10;

        var result = TextReversal.ReverseByLines(text, lineLength);
        System.Console.WriteLine("\nReversed text:");
        System.Console.WriteLine(result);
    }

    private static void HandleDiacriticalProcessing()
    {
        System.Console.Write("Enter text to process: ");
        var text = System.Console.ReadLine() ?? string.Empty;
        
        System.Console.WriteLine("Choose context:");
        System.Console.WriteLine("1. Default");
        System.Console.WriteLine("2. Emotional");
        System.Console.WriteLine("3. Technical");
        System.Console.WriteLine("4. Random");
        System.Console.Write("Enter choice (1-4): ");
        
        var contextChoice = System.Console.ReadLine();
        var context = contextChoice switch
        {
            "2" => DiacriticalProcessor.ProcessingContext.Emotional,
            "3" => DiacriticalProcessor.ProcessingContext.Technical,
            "4" => DiacriticalProcessor.ProcessingContext.Random,
            _ => DiacriticalProcessor.ProcessingContext.Default
        };

        var result = DiacriticalProcessor.ApplyAscentMarks(text, context);
        System.Console.WriteLine($"\nProcessed text ({context}):");
        System.Console.WriteLine(result);
    }

    private static void ShowPlatformInfo()
    {
        System.Console.WriteLine("=== Platform Information ===");
        System.Console.WriteLine(PlatformUtils.GetPlatformInfo());
        System.Console.WriteLine($"Current Platform: {PlatformUtils.GetCurrentPlatform()}");
        System.Console.WriteLine($"Is Android: {PlatformUtils.IsAndroid()}");
        System.Console.WriteLine($"Is HarmonyOS: {PlatformUtils.IsHarmonyOS()}");
        System.Console.WriteLine($"Unicode Support: {PlatformUtils.SupportsUnicode()}");
        System.Console.WriteLine($"Line Ending: {PlatformUtils.GetLineEnding().Replace("\r", "\\r").Replace("\n", "\\n")}");
        System.Console.WriteLine($"Path Separator: {PlatformUtils.GetPathSeparator()}");
        System.Console.WriteLine($"Temp Directory: {PlatformUtils.GetTempDirectory()}");
    }

    private static void RunDemo()
    {
        System.Console.WriteLine("=== Cross-Platform NLP Demo ===");
        
        // Demo text reversal with Chinese text
        var chineseText = "循环处理每一行：我们使用一个循环来处理每一行。对于每一行，我们计算出当前行的开始和结束索引，并提取出对应的子字符串。如果子字符串长度不足 n，则用空格填充";
        System.Console.WriteLine("Original Chinese text:");
        System.Console.WriteLine(chineseText);
        System.Console.WriteLine("\nReversed by lines (length 10):");
        System.Console.WriteLine(TextReversal.ReverseByLines(chineseText, 10));
        
        System.Console.WriteLine(new string('=', 50));
        
        // Demo diacritical processing
        var sampleText = "Hello world! This is a test of diacritical processing.";
        System.Console.WriteLine("Original text:");
        System.Console.WriteLine(sampleText);
        
        foreach (var context in Enum.GetValues<DiacriticalProcessor.ProcessingContext>())
        {
            var processed = DiacriticalProcessor.ApplyAscentMarks(sampleText, context, 42); // Fixed seed for reproducible results
            System.Console.WriteLine($"\n{context} context:");
            System.Console.WriteLine(processed);
        }
        
        System.Console.WriteLine(new string('=', 50));
        
        // Demo platform detection
        System.Console.WriteLine("Platform Detection:");
        ShowPlatformInfo();
    }

    private static void ShowHelp()
    {
        System.Console.WriteLine("NLP Note - Cross-Platform Text Processor");
        System.Console.WriteLine("Usage: nlpnote [command] [arguments]");
        System.Console.WriteLine();
        System.Console.WriteLine("Commands:");
        System.Console.WriteLine("  reverse <text> [lineLength]  - Reverse text by lines");
        System.Console.WriteLine("  diacritical <text> [context] - Apply diacritical marks");
        System.Console.WriteLine("  platform                     - Show platform information");
        System.Console.WriteLine("  demo                         - Run demonstration");
        System.Console.WriteLine();
        System.Console.WriteLine("If no command is provided, interactive mode will start.");
    }
}

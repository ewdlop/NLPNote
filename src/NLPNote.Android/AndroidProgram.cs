using NLPNote.Core;

namespace NLPNote.Android;

/// <summary>
/// Android-specific entry point for NLP Note application.
/// Optimized for Android platform with proper lifecycle management.
/// </summary>
public class AndroidProgram
{
    public static void Main(string[] args)
    {
        try
        {
            Console.WriteLine("=== NLP Note for Android ===");
            Console.WriteLine(PlatformUtils.GetPlatformInfo());
            
            // Verify Android platform detection
            if (PlatformUtils.IsAndroid())
            {
                Console.WriteLine("✓ Android platform detected successfully");
            }
            else
            {
                Console.WriteLine("⚠ Android platform detection may not be working properly");
            }
            
            Console.WriteLine();

            // Run Android-optimized demo
            RunAndroidDemo();
            
            // For Android, we'll run in command-line mode to avoid interactive input issues
            if (args.Length > 0)
            {
                ProcessAndroidCommands(args);
            }
            else
            {
                Console.WriteLine("Use command-line arguments for Android. Available commands:");
                Console.WriteLine("  demo - Run demonstration");
                Console.WriteLine("  platform - Show platform info");
                Console.WriteLine("  reverse <text> - Reverse text");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Environment.Exit(1);
        }
    }

    private static void RunAndroidDemo()
    {
        Console.WriteLine("=== Android NLP Processing Demo ===");
        
        // Test Chinese text processing on Android
        var chineseText = "安卓平台自然语言处理测试";
        Console.WriteLine($"Original: {chineseText}");
        Console.WriteLine($"Reversed: {TextReversal.ReverseString(chineseText)}");
        Console.WriteLine();
        
        // Test diacritical processing
        var testText = "Android Natural Language Processing";
        Console.WriteLine($"Original: {testText}");
        
        foreach (var context in Enum.GetValues<DiacriticalProcessor.ProcessingContext>())
        {
            var processed = DiacriticalProcessor.ApplyAscentMarks(testText, context, 42);
            Console.WriteLine($"{context}: {processed}");
        }
        
        Console.WriteLine();
        
        // Android-specific platform info
        Console.WriteLine("Android Platform Details:");
        Console.WriteLine($"Temp Directory: {PlatformUtils.GetTempDirectory()}");
        Console.WriteLine($"Path Separator: {PlatformUtils.GetPathSeparator()}");
        Console.WriteLine($"Line Ending: {PlatformUtils.GetLineEnding().Replace("\n", "\\n").Replace("\r", "\\r")}");
        Console.WriteLine();
    }

    private static void ProcessAndroidCommands(string[] args)
    {
        var command = args[0].ToLowerInvariant();
        
        switch (command)
        {
            case "demo":
                RunAndroidDemo();
                break;
                
            case "platform":
                Console.WriteLine(PlatformUtils.GetPlatformInfo());
                Console.WriteLine($"Android Detection: {PlatformUtils.IsAndroid()}");
                break;
                
            case "reverse":
                if (args.Length > 1)
                {
                    var result = TextReversal.ReverseString(args[1]);
                    Console.WriteLine($"Reversed: {result}");
                }
                else
                {
                    Console.WriteLine("Usage: reverse <text>");
                }
                break;
                
            case "diacritical":
                if (args.Length > 1)
                {
                    var context = args.Length > 2 && Enum.TryParse<DiacriticalProcessor.ProcessingContext>(args[2], true, out var ctx) 
                        ? ctx 
                        : DiacriticalProcessor.ProcessingContext.Default;
                    var result = DiacriticalProcessor.ApplyAscentMarks(args[1], context);
                    Console.WriteLine($"Processed: {result}");
                }
                else
                {
                    Console.WriteLine("Usage: diacritical <text> [context]");
                }
                break;
                
            default:
                Console.WriteLine($"Unknown command: {command}");
                break;
        }
    }
}
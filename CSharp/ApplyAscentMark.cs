using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    // Dictionary of diacritical marks mapped to their base letters
    static readonly Dictionary<char, string[]> DiacriticalMap = new Dictionary<char, string[]>()
    {
        {'a', new[] {"á", "à", "â", "ä", "ã"}},
        {'e', new[] {"é", "è", "ê", "ë"}},
        {'i', new[] {"í", "ì", "î", "ï"}},
        {'o', new[] {"ó", "ò", "ô", "ö", "õ"}},
        {'u', new[] {"ú", "ù", "û", "ü"}},
        {'c', new[] {"ç"}},
        {'n', new[] {"ñ"}},
        {'A', new[] {"Á", "À", "Â", "Ä", "Ã"}},
        {'E', new[] {"É", "È", "Ê", "Ë"}},
        {'I', new[] {"Í", "Ì", "Î", "Ï"}},
        {'O', new[] {"Ó", "Ò", "Ô", "Ö", "Õ"}},
        {'U', new[] {"Ú", "Ù", "Û", "Ü"}},
        {'C', new[] {"Ç"}},
        {'N', new[] {"Ñ"}}
    };

    static void Main(string[] args)
    {
        Console.WriteLine("Welcome to the Context-Aware Ascent Marked Converter!");
        Console.WriteLine("Enter a sentence to convert, and the context will guide the marking.");
        Console.WriteLine("Type 'exit' to quit.");

        while (true)
        {
            Console.Write("Enter your sentence: ");
            string input = Console.ReadLine();

            if (string.Equals(input, "exit", StringComparison.OrdinalIgnoreCase))
                break;

            string context = "Default";
            if (input.Contains("emotion"))
                context = "Emotional";
            else if (input.Contains("mechanic"))
                context = "Technical";

            string markedSentence = ApplyAscentMark(input, context);
            Console.WriteLine($"Context: {context}");
            Console.WriteLine($"Ascent Marked Sentence: {markedSentence}\n");
        }
    }

    static string ApplyAscentMark(string text, string context)
    {
        Random random = new Random();
        return new string(text.Select(c =>
        {
            if (DiacriticalMap.ContainsKey(c))
            {
                // Apply context-based diacritical selection
                string[] options = DiacriticalMap[c];
                return context switch
                {
                    "Emotional" => options.First(),  // Use the first variant for emotional context
                    "Technical" => options.Last(),   // Use the last variant for technical context
                    _ => options[random.Next(options.Length)] // Random for default context
                }[0];
            }
            return c; // Leave characters without diacritical marks unchanged
        }).ToArray());
    }
}

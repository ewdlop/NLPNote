using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

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

    // English language specific corrections
    static readonly Dictionary<string, string> EnglishCorrections = new Dictionary<string, string>()
    {
        {"teh", "the"},
        {"adn", "and"},
        {"recieve", "receive"},
        {"seperate", "separate"},
        {"definately", "definitely"},
        {"occured", "occurred"},
        {"alot", "a lot"},
        {"its", "it's"},  // Context dependent
        {"there", "their"}, // Context dependent
        {"your", "you're"}, // Context dependent
    };

    // Grammar patterns for English
    static readonly Dictionary<string, string> EnglishGrammarPatterns = new Dictionary<string, string>()
    {
        {@"\ba ([aeiouAEIOU])", "an $1"},  // a -> an before vowels
        {@"\ban ([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])", "a $1"},  // an -> a before consonants
        {@"\bI is\b", "I am"},  // Subject-verb agreement
        {@"\bYou is\b", "You are"},
        {@"\bWe is\b", "We are"},
        {@"\bThey is\b", "They are"},
    };

    static void Main(string[] args)
    {
        Console.WriteLine("Welcome to the Enhanced Context-Aware Ascent Marked Converter with English Patching!");
        Console.WriteLine("Enter a sentence to convert, and the context will guide the marking and corrections.");
        Console.WriteLine("Available contexts: Default, Emotional, Technical, English");
        Console.WriteLine("Type 'exit' to quit.");

        while (true)
        {
            Console.Write("Enter your sentence: ");
            string input = Console.ReadLine();

            if (string.Equals(input, "exit", StringComparison.OrdinalIgnoreCase))
                break;

            string context = DetermineContext(input);
            Console.WriteLine($"Detected Context: {context}");

            string processedSentence;
            if (context == "English" || ContainsEnglishText(input))
            {
                processedSentence = PatchEnglishText(input);
                Console.WriteLine($"English Patched: {processedSentence}");
                
                // Then apply accent marks if needed
                string markedSentence = ApplyAscentMark(processedSentence, context);
                Console.WriteLine($"Final Result: {markedSentence}");
            }
            else
            {
                processedSentence = ApplyAscentMark(input, context);
                Console.WriteLine($"Accent Marked: {processedSentence}");
            }
            
            Console.WriteLine();
        }
    }

    static string DetermineContext(string input)
    {
        if (input.Contains("emotion") || ContainsEmotionalWords(input))
            return "Emotional";
        else if (input.Contains("mechanic") || ContainsTechnicalWords(input))
            return "Technical";
        else if (ContainsEnglishText(input))
            return "English";
        else
            return "Default";
    }

    static bool ContainsEnglishText(string text)
    {
        // Simple heuristic: check for common English words
        string[] commonEnglishWords = { "the", "and", "is", "are", "was", "were", "have", "has", "will", "would", "could", "should" };
        string lowerText = text.ToLower();
        return commonEnglishWords.Any(word => lowerText.Contains($" {word} ") || lowerText.StartsWith($"{word} ") || lowerText.EndsWith($" {word}"));
    }

    static bool ContainsEmotionalWords(string text)
    {
        string[] emotionalWords = { "love", "hate", "happy", "sad", "angry", "excited", "feel", "emotion" };
        string lowerText = text.ToLower();
        return emotionalWords.Any(word => lowerText.Contains(word));
    }

    static bool ContainsTechnicalWords(string text)
    {
        string[] technicalWords = { "system", "process", "algorithm", "function", "method", "technical", "mechanic", "engine" };
        string lowerText = text.ToLower();
        return technicalWords.Any(word => lowerText.Contains(word));
    }

    static string PatchEnglishText(string text)
    {
        string patchedText = text;
        
        // Apply word-level corrections
        foreach (var correction in EnglishCorrections)
        {
            // Use word boundaries to avoid partial matches
            string pattern = $@"\b{Regex.Escape(correction.Key)}\b";
            patchedText = Regex.Replace(patchedText, pattern, correction.Value, RegexOptions.IgnoreCase);
        }

        // Apply grammar pattern corrections
        foreach (var pattern in EnglishGrammarPatterns)
        {
            patchedText = Regex.Replace(patchedText, pattern.Key, pattern.Value, RegexOptions.IgnoreCase);
        }

        // Fix spacing issues
        patchedText = Regex.Replace(patchedText, @"\s+", " "); // Multiple spaces to single space
        patchedText = Regex.Replace(patchedText, @"\s+([.!?,:;])", "$1"); // Remove space before punctuation
        patchedText = Regex.Replace(patchedText, @"([.!?,:;])([A-Za-z])", "$1 $2"); // Add space after punctuation

        // Capitalize first letter of sentences
        patchedText = Regex.Replace(patchedText, @"(^|[.!?]\s+)([a-z])", 
            match => match.Groups[1].Value + match.Groups[2].Value.ToUpper());

        return patchedText.Trim();
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
                char result = context switch
                {
                    "Emotional" => options.First()[0],  // Use the first variant for emotional context
                    "Technical" => options.Last()[0],   // Use the last variant for technical context
                    "English" => c, // Don't apply diacriticals to English text
                    _ => options[random.Next(options.Length)][0] // Random for default context
                };
                return result;
            }
            return c; // Leave characters without diacritical marks unchanged
        }).ToArray());
    }

    // Additional method to demonstrate English patching capabilities
    static void DemonstrateEnglishPatching()
    {
        string[] testSentences = {
            "teh quick brown fox jumps over the lazy dog",
            "I is going to the store",
            "this is a example of an sentence",
            "recieve the package seperate ly",
            "alot of people make mistakes"
        };

        Console.WriteLine("English Patching Demonstration:");
        Console.WriteLine("================================");

        foreach (string sentence in testSentences)
        {
            string patched = PatchEnglishText(sentence);
            Console.WriteLine($"Original: {sentence}");
            Console.WriteLine($"Patched:  {patched}");
            Console.WriteLine();
        }
    }
}

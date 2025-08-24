using System.Collections.Generic;

namespace NLPNote.Core;

/// <summary>
/// Provides cross-platform diacritical mark processing for multiple languages.
/// Supports Windows, Linux, macOS, Android, and HarmonyOS.
/// </summary>
public static class DiacriticalProcessor
{
    // Dictionary of diacritical marks mapped to their base letters
    private static readonly Dictionary<char, string[]> DiacriticalMap = new()
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

    /// <summary>
    /// Processing context for diacritical mark application
    /// </summary>
    public enum ProcessingContext
    {
        /// <summary>Default processing context</summary>
        Default,
        /// <summary>Emotional processing context with accent priority</summary>
        Emotional,
        /// <summary>Technical processing context with precision priority</summary>
        Technical,
        /// <summary>Random processing context for variety</summary>
        Random
    }

    /// <summary>
    /// Applies context-aware ascent marks to text with cross-platform support.
    /// </summary>
    /// <param name="text">The input text to process</param>
    /// <param name="context">The processing context</param>
    /// <param name="randomSeed">Optional random seed for reproducible results</param>
    /// <returns>Text with applied diacritical marks</returns>
    public static string ApplyAscentMarks(string text, ProcessingContext context = ProcessingContext.Default, int? randomSeed = null)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        var random = randomSeed.HasValue ? new Random(randomSeed.Value) : new Random();
        
        return new string(text.Select(c =>
        {
            if (DiacriticalMap.TryGetValue(c, out string[]? options))
            {
                return context switch
                {
                    ProcessingContext.Emotional => options.First()[0],  // Use the first variant for emotional context
                    ProcessingContext.Technical => options.Last()[0],   // Use the last variant for technical context
                    ProcessingContext.Random => options[random.Next(options.Length)][0], // Random selection
                    _ => options[random.Next(options.Length)][0] // Default to random
                };
            }
            return c; // Leave characters without diacritical marks unchanged
        }).ToArray());
    }

    /// <summary>
    /// Gets available diacritical options for a character.
    /// </summary>
    /// <param name="character">The base character</param>
    /// <returns>Array of diacritical variants, or empty array if none available</returns>
    public static string[] GetDiacriticalOptions(char character)
    {
        return DiacriticalMap.TryGetValue(character, out string[]? options) ? options : Array.Empty<string>();
    }

    /// <summary>
    /// Checks if a character has diacritical variants available.
    /// </summary>
    /// <param name="character">The character to check</param>
    /// <returns>True if diacritical variants are available</returns>
    public static bool HasDiacriticalVariants(char character)
    {
        return DiacriticalMap.ContainsKey(character);
    }

    /// <summary>
    /// Processes text for specific language contexts with platform-aware handling.
    /// </summary>
    /// <param name="text">Input text</param>
    /// <param name="languageCode">Language code (e.g., "es", "fr", "de")</param>
    /// <returns>Processed text with language-appropriate diacriticals</returns>
    public static string ProcessForLanguage(string text, string languageCode)
    {
        var context = languageCode?.ToLowerInvariant() switch
        {
            "es" or "spanish" => ProcessingContext.Emotional,
            "fr" or "french" => ProcessingContext.Technical,
            "de" or "german" => ProcessingContext.Technical,
            _ => ProcessingContext.Default
        };

        return ApplyAscentMarks(text, context);
    }
}
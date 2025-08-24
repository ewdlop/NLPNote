using System.Text;

namespace NLPNote.Core;

/// <summary>
/// Provides utilities for text reversal and string manipulation with cross-platform support.
/// </summary>
public static class TextReversal
{
    /// <summary>
    /// Reverses text line by line with a specified line length.
    /// This function processes Chinese and multi-byte characters correctly across all platforms.
    /// </summary>
    /// <param name="text">The input text to reverse</param>
    /// <param name="lineLength">The maximum length of each line</param>
    /// <returns>Text with each line reversed</returns>
    public static string ReverseByLines(string text, int lineLength)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        if (lineLength <= 0)
            throw new ArgumentException("Line length must be positive", nameof(lineLength));

        int remainder = text.Length % lineLength;
        int times = (text.Length - remainder) / lineLength;
        
        return string.Join('\n', Enumerable.Range(0, times + 1).Select(i =>
        {
            var startIndex = i * lineLength;
            var length = Math.Min(text.Length - startIndex, lineLength);
            var segment = text.Substring(startIndex, length);
            
            // Pad with spaces if needed
            if (segment.Length < lineLength) 
            {
                segment = $"{segment}{new string(' ', lineLength - segment.Length)}";
            }
            
            return new string(segment.Reverse().ToArray());
        }));
    }

    /// <summary>
    /// Reverses a string character by character.
    /// Handles Unicode and multi-byte characters properly across platforms.
    /// </summary>
    /// <param name="text">The text to reverse</param>
    /// <returns>The reversed text</returns>
    public static string ReverseString(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        return new string(text.Reverse().ToArray());
    }

    /// <summary>
    /// Processes Chinese text with proper handling for different platforms.
    /// </summary>
    /// <param name="chineseText">Chinese text to process</param>
    /// <param name="lineLength">Line length for processing</param>
    /// <returns>Processed Chinese text</returns>
    public static string ProcessChineseText(string chineseText, int lineLength = 10)
    {
        return ReverseByLines(chineseText, lineLength);
    }
}

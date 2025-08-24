namespace NLPNote.Core;

/// <summary>
/// Cross-platform utilities for detecting system information and platform-specific optimizations.
/// </summary>
public static class PlatformUtils
{
    /// <summary>
    /// Supported platform types
    /// </summary>
    public enum PlatformType
    {
        /// <summary>Microsoft Windows platform</summary>
        Windows,
        /// <summary>Linux-based platforms</summary>
        Linux,
        /// <summary>Apple macOS platform</summary>
        macOS,
        /// <summary>Google Android platform</summary>
        Android,
        /// <summary>Huawei HarmonyOS platform</summary>
        HarmonyOS,
        /// <summary>Unknown or unsupported platform</summary>
        Unknown
    }

    /// <summary>
    /// Detects the current platform with support for HarmonyOS detection.
    /// </summary>
    /// <returns>The detected platform type</returns>
    public static PlatformType GetCurrentPlatform()
    {
        if (OperatingSystem.IsWindows())
            return PlatformType.Windows;
        
        if (OperatingSystem.IsLinux())
        {
            // Check for Android
            if (IsAndroid())
                return PlatformType.Android;
            
            // Check for HarmonyOS (OpenHarmony)
            if (IsHarmonyOS())
                return PlatformType.HarmonyOS;
            
            return PlatformType.Linux;
        }
        
        if (OperatingSystem.IsMacOS())
            return PlatformType.macOS;
        
        return PlatformType.Unknown;
    }

    /// <summary>
    /// Checks if running on Android platform.
    /// </summary>
    /// <returns>True if running on Android</returns>
    public static bool IsAndroid()
    {
        try
        {
            // Android detection - check for Android-specific environment variables or properties
            var androidRoot = Environment.GetEnvironmentVariable("ANDROID_ROOT");
            var androidData = Environment.GetEnvironmentVariable("ANDROID_DATA");
            
            return !string.IsNullOrEmpty(androidRoot) || !string.IsNullOrEmpty(androidData);
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Checks if running on HarmonyOS platform.
    /// </summary>
    /// <returns>True if running on HarmonyOS</returns>
    public static bool IsHarmonyOS()
    {
        try
        {
            // HarmonyOS detection - check for HarmonyOS-specific indicators
            var osVersion = Environment.OSVersion.VersionString;
            
            // Check for HarmonyOS specific environment variables or system properties
            var harmonyIndicators = new[]
            {
                Environment.GetEnvironmentVariable("HARMONY_HOME"),
                Environment.GetEnvironmentVariable("OHOS_SDK_HOME"),
            };

            if (harmonyIndicators.Any(indicator => !string.IsNullOrEmpty(indicator)))
                return true;

            // Check OS version string for HarmonyOS indicators
            if (osVersion.Contains("HarmonyOS", StringComparison.OrdinalIgnoreCase) ||
                osVersion.Contains("OpenHarmony", StringComparison.OrdinalIgnoreCase) ||
                osVersion.Contains("OHOS", StringComparison.OrdinalIgnoreCase))
                return true;

            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets platform-specific line ending.
    /// </summary>
    /// <returns>Platform-appropriate line ending</returns>
    public static string GetLineEnding()
    {
        return GetCurrentPlatform() switch
        {
            PlatformType.Windows => "\r\n",
            PlatformType.Linux => "\n",
            PlatformType.macOS => "\n",
            PlatformType.Android => "\n",
            PlatformType.HarmonyOS => "\n",
            _ => Environment.NewLine
        };
    }

    /// <summary>
    /// Gets platform-specific path separator.
    /// </summary>
    /// <returns>Platform-appropriate path separator</returns>
    public static char GetPathSeparator()
    {
        return GetCurrentPlatform() switch
        {
            PlatformType.Windows => '\\',
            PlatformType.Linux => '/',
            PlatformType.macOS => '/',
            PlatformType.Android => '/',
            PlatformType.HarmonyOS => '/',
            _ => Path.DirectorySeparatorChar
        };
    }

    /// <summary>
    /// Gets platform-specific temporary directory.
    /// </summary>
    /// <returns>Platform-appropriate temporary directory path</returns>
    public static string GetTempDirectory()
    {
        return GetCurrentPlatform() switch
        {
            PlatformType.Android => "/data/local/tmp",
            PlatformType.HarmonyOS => "/tmp",
            _ => Path.GetTempPath()
        };
    }

    /// <summary>
    /// Checks if the current platform supports Unicode properly.
    /// </summary>
    /// <returns>True if Unicode is fully supported</returns>
    public static bool SupportsUnicode()
    {
        // All target platforms support Unicode in .NET 8
        return true;
    }

    /// <summary>
    /// Gets platform information as a formatted string.
    /// </summary>
    /// <returns>Formatted platform information</returns>
    public static string GetPlatformInfo()
    {
        var platform = GetCurrentPlatform();
        var architecture = Environment.Is64BitOperatingSystem ? "64-bit" : "32-bit";
        var framework = Environment.Version.ToString();
        
        return $"Platform: {platform} ({architecture}), .NET: {framework}";
    }
}
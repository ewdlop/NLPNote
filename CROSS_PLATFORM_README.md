# NLP Note - Cross-Platform Natural Language Processing

A cross-platform .NET 8 application for natural language processing that runs on Windows, Linux, macOS, Android, and HarmonyOS (鸿蒙).

## Features

- **Cross-Platform Support**: Runs on Windows, Linux, macOS, Android, and HarmonyOS
- **Text Reversal**: Advanced text reversal with proper Unicode and multi-byte character support
- **Diacritical Processing**: Context-aware diacritical mark application for multiple languages
- **Platform Detection**: Automatic detection of runtime platform including HarmonyOS
- **Multilingual Support**: Special optimizations for Chinese, English, and other languages
- **Interactive and Command-Line Modes**: Flexible usage options

## Supported Platforms

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Windows | x64, ARM64 | ✅ Supported | Native .NET support |
| Linux | x64, ARM64 | ✅ Supported | Native .NET support |
| macOS | x64, ARM64 (Apple Silicon) | ✅ Supported | Native .NET support |
| Android | ARM64, x64 | ✅ Supported | .NET Android support |
| HarmonyOS | ARM64, x64 | ✅ Supported | Linux-based runtime |

## Prerequisites

- .NET 8.0 SDK or later
- For Android: Android SDK (optional)
- For HarmonyOS: Compatible with OpenHarmony 3.0+

## Quick Start

### Building from Source

#### Using the Build Scripts

**On Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
```

**On Windows:**
```cmd
build.bat
```

#### Manual Build

```bash
# Restore dependencies
dotnet restore

# Build all projects
dotnet build -c Release

# Publish for specific platform
dotnet publish src/NLPNote.Console/NLPNote.Console.csproj -c Release -r linux-x64 -o dist/linux-x64
```

### Running the Application

#### Interactive Mode
```bash
# Windows
./dist/win-x64/nlpnote.exe

# Linux
./dist/linux-x64/nlpnote

# macOS
./dist/osx-x64/nlpnote

# HarmonyOS
./dist/harmonyos/nlpnote-harmonyos
```

#### Command-Line Mode
```bash
# Text reversal
./nlpnote reverse "Hello World" 5

# Diacritical processing
./nlpnote diacritical "Hello World" emotional

# Platform information
./nlpnote platform

# Run demo
./nlpnote demo
```

## Project Structure

```
NLPNote/
├── src/
│   ├── NLPNote.Core/           # Core cross-platform library
│   ├── NLPNote.Console/        # Console application (Windows/Linux/macOS)
│   ├── NLPNote.Android/        # Android-specific application
│   └── NLPNote.HarmonyOS/      # HarmonyOS-specific application
├── dist/                       # Build output directory
├── build.sh                    # Linux/macOS build script
├── build.bat                   # Windows build script
├── NLPNote.sln                 # Solution file
└── README.md                   # This file
```

## Core Features

### Text Reversal

The `TextReversal` class provides advanced text reversal capabilities:

```csharp
// Simple string reversal
string reversed = TextReversal.ReverseString("Hello World");

// Line-by-line reversal with specified length
string result = TextReversal.ReverseByLines(text, lineLength: 10);

// Chinese text processing
string processed = TextReversal.ProcessChineseText("你好世界", 5);
```

### Diacritical Processing

The `DiacriticalProcessor` class handles context-aware diacritical marks:

```csharp
// Apply diacritical marks with context
string result = DiacriticalProcessor.ApplyAscentMarks(
    "Hello World", 
    DiacriticalProcessor.ProcessingContext.Emotional
);

// Language-specific processing
string processed = DiacriticalProcessor.ProcessForLanguage(text, "spanish");
```

### Platform Detection

The `PlatformUtils` class provides comprehensive platform detection:

```csharp
// Detect current platform
var platform = PlatformUtils.GetCurrentPlatform();

// Check for specific platforms
bool isAndroid = PlatformUtils.IsAndroid();
bool isHarmonyOS = PlatformUtils.IsHarmonyOS();

// Get platform-specific information
string lineEnding = PlatformUtils.GetLineEnding();
char pathSeparator = PlatformUtils.GetPathSeparator();
```

## Platform-Specific Features

### Android Support

- Optimized for Android lifecycle
- Proper Unicode handling on Android
- Android-specific temporary directory detection

### HarmonyOS Support

- Detection of HarmonyOS/OpenHarmony environment
- Support for HarmonyOS environment variables
- Bilingual interface (English/Chinese)
- Compatible with HarmonyOS 3.0+

### Windows Support

- Native Windows .NET support
- Windows-specific path handling
- Support for both x64 and ARM64 architectures

### macOS Support

- Native macOS support including Apple Silicon (ARM64)
- Proper macOS path conventions
- Universal binary support

### Linux Support

- Broad Linux distribution compatibility
- ARM64 support for devices like Raspberry Pi
- Container-friendly deployment

## Development

### Adding New Platforms

To add support for a new platform:

1. Update `PlatformUtils.cs` with platform detection logic
2. Create platform-specific project if needed
3. Add runtime identifier to project files
4. Update build scripts
5. Add platform-specific optimizations

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on multiple platforms
5. Submit a pull request

## Runtime Identifiers (RIDs)

The following Runtime Identifiers are supported:

- `win-x64` - Windows 64-bit
- `win-arm64` - Windows ARM64
- `linux-x64` - Linux 64-bit
- `linux-arm64` - Linux ARM64
- `osx-x64` - macOS 64-bit (Intel)
- `osx-arm64` - macOS ARM64 (Apple Silicon)
- `android-arm64` - Android ARM64
- `android-x64` - Android x64

## Environment Variables

### HarmonyOS Detection

The application recognizes these HarmonyOS-specific environment variables:

- `HARMONY_HOME` - HarmonyOS SDK home directory
- `OHOS_SDK_HOME` - OpenHarmony SDK home directory

### Android Detection

The application recognizes these Android-specific environment variables:

- `ANDROID_ROOT` - Android system root
- `ANDROID_DATA` - Android data directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- .NET Team for excellent cross-platform support
- HarmonyOS team for platform documentation
- Contributors to the NLP Note project

## Support

For issues and questions:
1. Check the [Issues](https://github.com/ewdlop/NLPNote/issues) page
2. Create a new issue with platform-specific details
3. Include .NET version and platform information

---

**Made with ❤️ for cross-platform natural language processing**
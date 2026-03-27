# apfel — Project Instructions

## The Golden Goal

apfel has ONE purpose with FOUR delivery modes:

> **Expose Apple's on-device FoundationModels LLM as a usable, powerful UNIX tool
> and an OpenAI API-compatible server. Bonus: debuggable via native GUI, working
> command-line chat.**

### The four modes, in priority order:

1. **UNIX tool** (`apfel "prompt"`, `echo "text" | apfel`, `apfel --stream`)
   - Pipe-friendly, composable, correct exit codes
   - Works with `jq`, `xargs`, shell scripts
   - `--json` output for machine consumption
   - Respects `NO_COLOR`, `--quiet`, stdin detection

2. **OpenAI-compatible HTTP server** (`apfel --serve`)
   - Drop-in replacement for `openai.OpenAI(base_url="http://localhost:11434/v1")`
   - `/v1/chat/completions` (streaming + non-streaming)
   - `/v1/models`, `/health`, tool calling, `response_format`
   - Honest 501s for unsupported features (embeddings, legacy completions)
   - CORS for browser clients

3. **Command-line chat** (`apfel --chat`)
   - Interactive multi-turn with context window protection
   - Typed error display, context rotation when approaching limit
   - System prompt support

4. **Debug GUI** (`apfel --gui`)
   - Native SwiftUI inspector: request/response JSON, curl commands, logs
   - Talks to `--serve` via HTTP (pure consumer, no model logic)
   - TTS, STT, self-discussion mode

### Non-negotiable principles:

- **100% on-device.** No cloud, no API keys, no network for inference. Ever.
- **Honest about limitations.** 4096 token context, no embeddings, no vision — say so clearly.
- **Clean code, clean logic.** No hacks. Proper error types. Real token counts.
- **Swift 6 strict concurrency.** No data races.

## Architecture

```
CLI (single/stream/chat) ──┐
                           ├─→ Session.swift → FoundationModels (on-device)
HTTP Server (/v1/*) ───────┤
                           ├─→ ContextManager → Transcript API
GUI (SwiftUI) ─── HTTP ────┘   SchemaConverter → DynamicGenerationSchema
                                TokenCounter → real tokenCount (SDK 26.4)
```

- `ApfelCore` library: pure Swift, no FoundationModels dependency, unit-testable
- Main target: FoundationModels integration, Hummingbird HTTP server
- Tests: `swift run apfel-tests` (pure Swift runner, no XCTest needed)

## Build & Test

```bash
make install                   # build release + install to /usr/local/bin
swift build                    # debug build only
swift run apfel-tests          # run pure Swift unit tests
apfel "Hello"                  # single prompt (after make install)
apfel --serve                  # start server on :11434
```

**Always use `make install` for testing changes** - `swift run` uses a debug build, and the installed binary at `/usr/local/bin/apfel` won't reflect your changes until you run `make install`.

Integration tests (requires server running):
```bash
python3 -m pytest Tests/integration/ -v    # release-binary integration tests
```

## Key Files

| Area | Files |
|------|-------|
| Entry point | `Sources/main.swift` |
| CLI commands | `Sources/CLI.swift` |
| HTTP server | `Sources/Server.swift`, `Sources/Handlers.swift` |
| Session mgmt | `Sources/Session.swift`, `Sources/ContextManager.swift` |
| Tool calling | `Sources/Core/ToolCallHandler.swift`, `Sources/SchemaConverter.swift` |
| Token counting | `Sources/TokenCounter.swift` |
| Error types | `Sources/Core/ApfelError.swift` |
| Models/types | `Sources/Models.swift`, `Sources/ToolModels.swift` |
| GUI | `Sources/GUI/` (SwiftUI, talks to server via HTTP) |
| Tests | `Tests/apfelTests/` |
| Tickets | `open-tickets/` |

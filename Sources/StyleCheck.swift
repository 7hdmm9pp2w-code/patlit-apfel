// ============================================================================
// StyleCheck.swift — patlit Tone & Style Guide checker (CLI mode)
// patlit-ai fork of apfel
//
// Usage:
//   patlit-ai --style-check --channel email "Text..."
//   echo "Draft" | patlit-ai --style-check --channel schriftsatz
//   patlit-ai --style-check --channel linkedin --backend cloud < post.txt
// ============================================================================

import Foundation
import FoundationModels

// MARK: - Channel

enum PatliAIStyleChannel: String {
    case website     = "website"
    case linkedin    = "linkedin"
    case email       = "email"
    case schriftsatz = "schriftsatz"

    var description: String {
        switch self {
        case .website:     return "Website (Englisch, max. 20 Wörter/Satz)"
        case .linkedin:    return "LinkedIn (Englisch oder Deutsch, ein Gedanke)"
        case .email:       return "E-Mail Mandant (kein Kanzleideutsch)"
        case .schriftsatz: return "Schriftsatz (Deutsch, aktiv, präzise)"
        }
    }

    static func from(_ raw: String) -> PatliAIStyleChannel {
        PatliAIStyleChannel(rawValue: raw.lowercased()) ?? .email
    }
}

// MARK: - Style Guide System Prompt

enum PatliAIStyleGuide {

    static func systemPrompt(channel: PatliAIStyleChannel) -> String {
        """
        Du bist der Stilprüfer für die Kanzlei patlit.xyz.
        Prüfe den vorgelegten Text gegen den patlit Tone & Style Guide v1.0.
        Antworte in der Sprache des eingereichten Textes (Deutsch oder Englisch).
        Aktueller Kanal: \(channel.rawValue.uppercased()) — \(channel.description)

        ## VIER GRUNDPRINZIPIEN

        1. PRÄZISE — Ein Satz, ein Gedanke. Kein Hedging. Direkt zum Punkt.
        2. DIREKT — Das eigentliche Problem ansprechen. Dinge beim Namen nennen.
        3. SELBSTSICHER — Positionen, keine Möglichkeiten.
           "Wir widersprechen" statt "es könnte argumentiert werden".
        4. TROCKEN — Kein Enthusiasmus. Keine Ausrufezeichen. Keine Emojis.

        ## SPRACHE
        Ausschließlich Englisch ODER Deutsch. Niemals gemischt.
        Fachbegriffe (UPC, EPO, prior art, wortsinngemäß) dürfen unübersetzt bleiben.

        ## VERBOTENE WÖRTER
        Deutsch: ganzheitlich, maßgeschneidert, auf Augenhöhe, kompetent, erfahren,
        partnerschaftlich, "Ich freue mich mitteilen zu dürfen", "Wir bieten ... Lösungen"
        Englisch: innovative, disrupting, solutions, synergies, value-add,
        "excited to share", "we'd love to connect", "explore potential synergies"

        ## BEVORZUGT
        Englisch: senior counsel, outcome, precision, "leverage technology",
        "we disagree", "no [X]", prior art
        Deutsch: wortsinngemäß, Streitpatent, "Wir widersprechen",
        "Die angegriffene Ausführungsform verwirklicht..."

        ## AUSGABEFORMAT — strikt einhalten

        ### Bewertung
        PASS | PASS MIT ANMERKUNGEN | ÜBERARBEITUNG ERFORDERLICH
        [Ein Satz Begründung]

        ### Verstöße
        - **[Regelbereich]** „Zitat" → Erklärung → Vorschlag
        [Nur wenn Verstöße vorhanden]

        ### Überarbeiteter Text
        [Vollständig korrigierte Version — nur bei ÜBERARBEITUNG ERFORDERLICH]

        Bei PASS: nur die Bewertungszeile. Kein Lob. Kein weiterer Text.
        """
    }
}

// MARK: - Runner

struct PatliAIStyleCheckRunner {

    let text: String
    let channelRaw: String
    let backendRaw: String
    let stream: Bool

    func run() async throws {
        let channel = PatliAIStyleChannel.from(channelRaw)
        let systemPrompt = PatliAIStyleGuide.systemPrompt(channel: channel)
        let userPrompt = "Kanal: \(channel.rawValue)\n\nText zur Prüfung:\n---\n\(text)\n---"

        switch backendRaw {

        case "medium":
            guard await MLXProxy.isReachable() else {
                fputs("Fehler: \(PatliAIBackendError.mlxUnavailable.description)\n", stderr)
                return
            }
            let msgs: [[String: String]] = [
                ["role": "system", "content": systemPrompt],
                ["role": "user",   "content": userPrompt]
            ]
            let bodyObj: [String: Any] = [
                "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "messages": msgs,
                "max_tokens": 2048,
                "stream": stream
            ]
            let bodyData = try JSONSerialization.data(withJSONObject: bodyObj)

            if stream {
                for try await line in MLXProxy.streamLines(body: bodyData) {
                    guard line.hasPrefix("data: ") else { continue }
                    let json = String(line.dropFirst(6))
                    guard json != "[DONE]",
                          let data = json.data(using: .utf8),
                          let obj  = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let choices = obj["choices"] as? [[String: Any]],
                          let delta   = choices.first?["delta"] as? [String: Any],
                          let text    = delta["content"] as? String else { continue }
                    print(text, terminator: "")
                    fflush(stdout)
                }
                print()
            } else {
                let (data, _) = try await MLXProxy.forwardRaw(body: bodyData)
                if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = obj["choices"] as? [[String: Any]],
                   let msg     = choices.first?["message"] as? [String: Any],
                   let content = msg["content"] as? String {
                    print(content)
                }
            }

        case "cloud":
            guard let apiKey = PatliAIKeychain.load(service: "patlit-ai",
                                                    account: "anthropic-api-key"),
                  !apiKey.isEmpty else {
                fputs("Fehler: \(PatliAIBackendError.missingAPIKey.description)\n", stderr)
                return
            }
            let msgs = [["role": "user", "content": userPrompt]]
            let bodyObj: [String: Any] = [
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "system": systemPrompt,
                "messages": msgs,
                "stream": stream
            ]
            let bodyData = try JSONSerialization.data(withJSONObject: bodyObj)
            var req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.setValue(apiKey,             forHTTPHeaderField: "x-api-key")
            req.setValue("2023-06-01",       forHTTPHeaderField: "anthropic-version")
            req.httpBody = bodyData

            if stream {
                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                for try await line in bytes.lines {
                    guard line.hasPrefix("data: ") else { continue }
                    let json = String(line.dropFirst(6))
                    guard json != "[DONE]",
                          let data  = json.data(using: .utf8),
                          let obj   = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let type  = (obj["type"] as? String), type == "content_block_delta",
                          let delta = obj["delta"] as? [String: Any],
                          let text  = delta["text"] as? String else { continue }
                    print(text, terminator: "")
                    fflush(stdout)
                }
                print()
            } else {
                let (data, _) = try await URLSession.shared.data(for: req)
                if let obj     = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let content = obj["content"] as? [[String: Any]],
                   let text    = content.first?["text"] as? String {
                    print(text)
                }
            }

        default: // "local" — Apple FoundationModels
            let session = LanguageModelSession()
            if stream {
                let responseStream = session.streamResponse(to: "\(systemPrompt)\n\n\(userPrompt)")
                for try await chunk in responseStream {
                    print(chunk, terminator: "")
                    fflush(stdout)
                }
                print()
            } else {
                let response = try await session.respond(to: "\(systemPrompt)\n\n\(userPrompt)")
                print(response.content)
            }
        }
    }
}

// MARK: - MLX streaming helper (used by CLI — server uses MLXProxy.forward)

extension MLXProxy {

    static func streamLines(body: Data) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    var req = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.httpBody = body
                    let (bytes, _) = try await URLSession.shared.bytes(for: req)
                    for try await line in bytes.lines { continuation.yield(line) }
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
        }
    }

    static func forwardRaw(body: Data) async throws -> (Data, URLResponse) {
        var req = URLRequest(url: baseURL.appendingPathComponent("v1/chat/completions"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        return try await URLSession.shared.data(for: req)
    }
}

// ============================================================================
// MultiBackend.swift — Multi-backend routing for patlit-ai
// patlit-ai fork of apfel (https://github.com/Arthur-Ficial/apfel)
//
// Adds MLX (Qwen) and Anthropic (Claude) backends alongside the existing
// Apple FoundationModels backend.
// ============================================================================

import Foundation
import Hummingbird
import NIOCore
import Security

// MARK: - Router
// Called from Handlers.swift for backend != "local"

enum PatliAIRouter {

    static func handle(
        chatRequest: ChatCompletionRequest,
        backend: String,
        isStreaming: Bool,
        requestBodyString: String?,
        events: [String]
    ) async throws -> (response: Response, trace: ChatRequestTrace) {

        switch backend {

        case "medium":
            guard await MLXProxy.isReachable() else {
                return errorResponse(
                    message: PatliAIBackendError.mlxUnavailable.description,
                    type: "server_error",
                    isStreaming: isStreaming,
                    requestBody: requestBodyString,
                    events: events
                )
            }
            return try await MLXProxy.forward(chatRequest: chatRequest,
                                              isStreaming: isStreaming,
                                              requestBody: requestBodyString,
                                              events: events)

        case "cloud":
            return try await AnthropicHandler.handle(chatRequest: chatRequest,
                                                     isStreaming: isStreaming,
                                                     requestBody: requestBodyString,
                                                     events: events)

        default:
            return errorResponse(
                message: "Unknown backend: \(backend). Use local, medium, or cloud.",
                type: "invalid_request_error",
                isStreaming: isStreaming,
                requestBody: requestBodyString,
                events: events
            )
        }
    }

    private static func errorResponse(
        message: String,
        type: String,
        isStreaming: Bool,
        requestBody: String?,
        events: [String]
    ) -> (response: Response, trace: ChatRequestTrace) {
        let body = """
        {"error":{"message":"\(message)","type":"\(type)"}}
        """
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return (
            Response(status: .serviceUnavailable,
                     headers: headers,
                     body: .init(byteBuffer: ByteBuffer(string: body))),
            ChatRequestTrace(stream: isStreaming, estimatedTokens: nil,
                             error: message, requestBody: requestBody,
                             responseBody: nil, events: events)
        )
    }
}

// MARK: - MLX Proxy
// Forwards requests verbatim to the MLX server (OpenAI-compatible)

enum MLXProxy {

    static let baseURL = URL(string: "http://127.0.0.1:8080")!
    static let defaultModel = "mlx-community/Qwen3.6-35B-A3B-4bit"

    static func isReachable() async -> Bool {
        let url = baseURL.appendingPathComponent("health")
        var req = URLRequest(url: url, timeoutInterval: 1.5)
        req.httpMethod = "GET"
        return (try? await URLSession.shared.data(for: req)) != nil
    }

    static func forward(
        chatRequest: ChatCompletionRequest,
        isStreaming: Bool,
        requestBody: String?,
        events: [String]
    ) async throws -> (response: Response, trace: ChatRequestTrace) {

        // Re-encode with MLX model name substituted
        var body = buildRequestBody(chatRequest: chatRequest, model: defaultModel)
        let url = baseURL.appendingPathComponent("v1/chat/completions")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: req)
        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 200

        var headers = HTTPFields()
        headers[.contentType] = isStreaming ? "text/event-stream" : "application/json"

        return (
            Response(status: .init(code: statusCode),
                     headers: headers,
                     body: .init(byteBuffer: ByteBuffer(data: data))),
            ChatRequestTrace(stream: isStreaming, estimatedTokens: nil, error: nil,
                             requestBody: requestBody, responseBody: nil, events: events)
        )
    }

    private static func buildRequestBody(chatRequest: ChatCompletionRequest, model: String) -> Data {
        // Build a minimal OpenAI-compatible body for MLX
        var obj: [String: Any] = [
            "model": model,
            "stream": chatRequest.stream ?? false,
            "messages": chatRequest.messages.map { m -> [String: String] in
                ["role": m.role, "content": m.textContent ?? ""]
            }
        ]
        if let t = chatRequest.temperature { obj["temperature"] = t }
        if let m = chatRequest.max_tokens { obj["max_tokens"] = m }
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }
}

// MARK: - Anthropic Handler
// Native Anthropic Messages API — not an OpenAI shim

enum AnthropicHandler {

    static func handle(
        chatRequest: ChatCompletionRequest,
        isStreaming: Bool,
        requestBody: String?,
        events: [String]
    ) async throws -> (response: Response, trace: ChatRequestTrace) {

        guard let apiKey = PatliAIKeychain.load(service: "patlit-ai",
                                                account: "anthropic-api-key"),
              !apiKey.isEmpty else {
            return PatliAIRouter.handle(
                chatRequest: chatRequest,
                backend: "_error",
                isStreaming: isStreaming,
                requestBodyString: requestBody,
                events: events
            ) as! (response: Response, trace: ChatRequestTrace)
            // Inline error — simpler:
        }

        // Extract system prompt and messages
        let systemText = chatRequest.messages
            .filter { $0.role == "system" }
            .compactMap { $0.textContent }
            .joined(separator: "\n")

        let msgs = chatRequest.messages
            .filter { $0.role != "system" }
            .compactMap { m -> [String: String]? in
                guard let text = m.textContent else { return nil }
                return ["role": m.role, "content": text]
            }

        // Build Anthropic request body
        var anthropicBody: [String: Any] = [
            "model": "claude-sonnet-4-20250514",
            "max_tokens": chatRequest.max_tokens ?? 4096,
            "messages": msgs,
            "stream": isStreaming
        ]
        if !systemText.isEmpty { anthropicBody["system"] = systemText }

        let bodyData = try JSONSerialization.data(withJSONObject: anthropicBody)
        var req = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
        req.httpMethod = "POST"
        req.setValue("application/json",   forHTTPHeaderField: "Content-Type")
        req.setValue(apiKey,               forHTTPHeaderField: "x-api-key")
        req.setValue("2023-06-01",         forHTTPHeaderField: "anthropic-version")
        req.httpBody = bodyData

        let (respData, httpResp) = try await URLSession.shared.data(for: req)
        let statusCode = (httpResp as? HTTPURLResponse)?.statusCode ?? 200

        if isStreaming {
            // Return Anthropic SSE directly — GUI/clients consuming this
            // endpoint should handle Anthropic SSE format
            var headers = HTTPFields()
            headers[.contentType] = "text/event-stream"
            return (
                Response(status: .init(code: statusCode),
                         headers: headers,
                         body: .init(byteBuffer: ByteBuffer(data: respData))),
                ChatRequestTrace(stream: true, estimatedTokens: nil, error: nil,
                                 requestBody: requestBody, responseBody: nil, events: events)
            )
        } else {
            // Convert Anthropic response to OpenAI format
            let openAIBody = convertToOpenAI(anthropicData: respData,
                                             model: chatRequest.model)
            var headers = HTTPFields()
            headers[.contentType] = "application/json"
            return (
                Response(status: .init(code: statusCode),
                         headers: headers,
                         body: .init(byteBuffer: ByteBuffer(data: openAIBody))),
                ChatRequestTrace(stream: false, estimatedTokens: nil, error: nil,
                                 requestBody: requestBody, responseBody: nil, events: events)
            )
        }
    }

    private static func convertToOpenAI(anthropicData: Data, model: String) -> Data {
        guard let json = try? JSONSerialization.jsonObject(with: anthropicData) as? [String: Any],
              let content = json["content"] as? [[String: Any]],
              let text = content.first?["text"] as? String else {
            return anthropicData // fallback
        }
        let openAI: [String: Any] = [
            "id": "chatcmpl-\(UUID().uuidString.prefix(12).lowercased())",
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [[
                "index": 0,
                "message": ["role": "assistant", "content": text],
                "finish_reason": json["stop_reason"] as? String ?? "stop"
            ]],
            "usage": ["prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0]
        ]
        return (try? JSONSerialization.data(withJSONObject: openAI)) ?? anthropicData
    }
}

// MARK: - Keychain

enum PatliAIKeychain {

    static func save(_ value: String, service: String, account: String) {
        guard let data = value.data(using: .utf8) else { return }
        let q: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String:   data
        ]
        SecItemDelete(q as CFDictionary)
        SecItemAdd(q as CFDictionary, nil)
    }

    static func load(service: String, account: String) -> String? {
        let q: [String: Any] = [
            kSecClass as String:        kSecClassGenericPassword,
            kSecAttrService as String:  service,
            kSecAttrAccount as String:  account,
            kSecReturnData as String:   true,
            kSecMatchLimit as String:   kSecMatchLimitOne
        ]
        var result: AnyObject?
        SecItemCopyMatching(q as CFDictionary, &result)
        guard let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - Errors

enum PatliAIBackendError: Error, CustomStringConvertible {
    case missingAPIKey
    case mlxUnavailable
    case httpError(Int, String)

    var description: String {
        switch self {
        case .missingAPIKey:
            return "Kein Anthropic API-Key. Setzen mit: patlit-ai --set-api-key sk-ant-..."
        case .mlxUnavailable:
            return "MLX-Server nicht erreichbar. Starte: mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080"
        case .httpError(let code, let body):
            return "HTTP \(code): \(body)"
        }
    }
}

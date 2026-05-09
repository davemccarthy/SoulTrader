import Foundation

/// Pipe segments, Article URLs, and label styling — same rules as holding discovery card.
enum DiscoveryExplanationFormatting {
    static func attributed(from raw: String?) -> AttributedString {
        let pieces = parseDiscoveryExplanationPieces(raw)
        guard !pieces.isEmpty else {
            return AttributedString("No discovery explanation available.")
        }
        var result = AttributedString()
        for (idx, piece) in pieces.enumerated() {
            if idx > 0 {
                result.append(AttributedString("\n\n"))
            }
            switch piece {
            case let .plain(s):
                result.append(styledDiscoveryPlainSegment(s))
            case let .articleLink(title, url):
                var linkText = AttributedString(title)
                linkText.link = url
                result.append(linkText)
            case let .bareURL(url):
                var linkText = AttributedString(url.absoluteString)
                linkText.link = url
                result.append(linkText)
            }
        }
        return result
    }

    private enum DiscoveryExplanationPiece {
        case plain(String)
        case articleLink(title: String, url: URL)
        case bareURL(URL)
    }

    private static func styledDiscoveryPlainSegment(_ segment: String) -> AttributedString {
        let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let regex = try? NSRegularExpression(pattern: #"^([A-Za-z][A-Za-z ]*):\s*(.*)$"#),
              let match = regex.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
              let labelRange = Range(match.range(at: 1), in: trimmed),
              let valueRange = Range(match.range(at: 2), in: trimmed) else {
            return AttributedString(trimmed)
        }

        let labelRaw = String(trimmed[labelRange])
        let value = String(trimmed[valueRange])

        // Hide stored "Article:" prefix in UI while keeping raw explanation intact for pairing logic.
        if labelRaw.lowercased() == "article" {
            return AttributedString(value.trimmingCharacters(in: .whitespacesAndNewlines))
        }

        let label = labelRaw.uppercased()
        var attributed = AttributedString()
        var labelText = AttributedString("\(label):")
        labelText.inlinePresentationIntent = .stronglyEmphasized
        attributed.append(labelText)
        if !value.isEmpty {
            attributed.append(AttributedString(" \(value)"))
        }
        return attributed
    }

    private static func parseDiscoveryExplanationPieces(_ raw: String?) -> [DiscoveryExplanationPiece] {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let normalized = raw.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return [] }
        let segments = normalized.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var pieces: [DiscoveryExplanationPiece] = []
        var i = 0
        while i < segments.count {
            let current = segments[i]
            if let title = discoveryArticleTitle(from: current), i + 1 < segments.count {
                let urlRaw = segments[i + 1]
                if urlRaw.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil,
                   let url = URL(string: urlRaw) {
                    pieces.append(.articleLink(title: title, url: url))
                    i += 2
                    continue
                }
            }
            if current.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil,
               let url = URL(string: current) {
                pieces.append(.bareURL(url))
                i += 1
                continue
            }
            pieces.append(.plain(current))
            i += 1
        }
        return pieces
    }

    private static func discoveryArticleTitle(from segment: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: "^article\\s*:\\s*(.+)$", options: .caseInsensitive) else { return nil }
        let range = NSRange(segment.startIndex..., in: segment)
        guard let match = regex.firstMatch(in: segment, range: range),
              let titleRange = Range(match.range(at: 1), in: segment) else { return nil }
        let title = String(segment[titleRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        return title.isEmpty ? nil : title
    }
}

import SwiftUI

/// Shared EDGAR Ex-99 / media / bonuses rendering (health history + discovery meta).
enum EdgarStructuredSections {
    @ViewBuilder
    static func content(
        ex99: HealthEx99Payload?,
        media: HealthMediaPayload?,
        bonuses: [String],
        penalties: [String]
    ) -> some View {
        Text("EDGAR Ex-99")
            .appStyle(.cardTitle)
        justificationsBody(ex99)
        ex99LabelRows(ex99)
        mediaBody(media)
        bonusesBody(bonuses: bonuses, penalties: penalties)
    }

    @ViewBuilder
    static func justificationsBody(_ ex99: HealthEx99Payload?) -> some View {
        let justificationPairs = edgarJustificationParagraphs(ex99?.justifications)
        if justificationPairs.isEmpty {
            Text("No EX-99 justifications.")
                .appStyle(.detailBodyMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            ForEach(Array(justificationPairs.enumerated()), id: \.offset) { _, pair in
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(pair.0):")
                        .appStyle(.detailFieldLabel)
                    Text(pair.1)
                        .detailBody()
                }
            }
        }
    }

    @ViewBuilder
    static func ex99LabelRows(_ ex: HealthEx99Payload?) -> some View {
        let topRows = edgarTopLevelEx99Rows(ex)
        if !topRows.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(topRows.enumerated()), id: \.offset) { _, row in
                    edgarKeyValueRow(label: row.0, value: row.1)
                }
            }
            .padding(.top, 6)
        }
    }

    @ViewBuilder
    static func mediaBody(_ media: HealthMediaPayload?) -> some View {
        if media?.hasStructuredContent == true {
            Text("EDGAR Media")
                .appStyle(.cardTitle)
                .padding(.top, 10)

            if let s = media?.summary?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty {
                Text(s)
                    .detailBody()
            }

            edgarKeyValueRow(label: "Sentiment", value: media?.sentiment)
            edgarKeyValueRow(label: "EPS", value: media?.eps)
            edgarKeyValueRow(label: "Revenue", value: media?.revenue)
            edgarKeyValueRow(label: "Broker", value: media?.broker)

            edgarBulletList(title: "Headlines", items: media?.headlines ?? [], maxItems: 4)
            edgarBulletList(title: "Red Flags", items: media?.redFlags ?? [], maxItems: 4)
        }
    }

    @ViewBuilder
    static func bonusesBody(bonuses: [String], penalties: [String]) -> some View {
        Text("Bonuses / Penalties")
            .appStyle(.cardTitle)
            .padding(.top, 10)

        if bonuses.isEmpty && penalties.isEmpty {
            Text("No bonuses or penalties.")
                .appStyle(.detailBodyMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            edgarBulletList(title: "Bonuses", items: bonuses, maxItems: 6)
            edgarBulletList(title: "Penalties", items: penalties, maxItems: 6)
        }
    }

    static func edgarJustificationParagraphs(_ j: HealthEx99Justifications?) -> [(String, String)] {
        guard let j else { return [] }
        let pairs: [(String, String?)] = [
            ("Past Performance", j.pastPerformance),
            ("Guidance", j.guidance),
            ("Expectation", j.expectation),
            ("Market Reaction", j.marketReaction),
        ]
        return pairs.compactMap { title, text in
            guard let t = text?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
            return (title, t)
        }
    }

    static func edgarTopLevelEx99Rows(_ ex: HealthEx99Payload?) -> [(String, String)] {
        guard let ex else { return [] }
        let pairs: [(String, String?)] = [
            ("Expectation", ex.expectation),
            ("Guidance", ex.guidance),
            ("Performance", ex.pastPerformance),
            ("Market", ex.marketReaction),
        ]
        return pairs.compactMap { label, v in
            guard let t = v?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
            return (label, t)
        }
    }

    @ViewBuilder
    static func edgarKeyValueRow(label: String, value: String?) -> some View {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty { EmptyView() } else {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\(label):")
                    .appStyle(.detailRowLabel)
                Text(trimmed)
                    .appStyle(.detailRowValue)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder
    static func edgarBulletList(title: String, items: [String], maxItems: Int) -> some View {
        let trimmed = items.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        if trimmed.isEmpty { EmptyView() } else {
            Text("\(title):")
                .appStyle(.detailFieldLabel)
                .padding(.top, 4)
            let shown = Array(trimmed.prefix(maxItems))
            ForEach(Array(shown.enumerated()), id: \.offset) { _, line in
                Text("• \(line)")
                    .appStyle(.detailBodyMuted)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if trimmed.count > shown.count {
                Text("… +\(trimmed.count - shown.count) more")
                    .appStyle(.listSubline)
            }
        }
    }
}

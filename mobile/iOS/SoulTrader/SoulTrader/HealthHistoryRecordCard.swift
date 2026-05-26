import SwiftUI

/// Health check panel — shared by holding detail and discovery detail (matches prior `HoldingDetailView.healthRecordCard`).
struct HealthHistoryRecordCard: View {
    let record: HealthHistoryRecord
    var checkNumber: Int? = nil

    var body: some View {
        let checkTitle = checkNumber.map { "Health check #\($0)" } ?? "Health check"
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(checkTitle)
                    .appStyle(.cardTitle)
                Spacer()
                Text(String(format: "Score %.1f", record.score))
                    .appStyle(.metricValue)
            }

            Text(formatHealthDate(record.created))
                .appStyle(.detailMeta)

            if record.renderKind == "edgar", record.hasEdgarStructuredPayload {
                edgarStructuredHealthSections()
            } else {
                advisorHealthSections()
            }
        }
        .cardSurface()
    }

    @ViewBuilder
    private func edgarStructuredHealthSections() -> some View {
        let ex = record.meta?.ex99
        let media = record.meta?.media
        let bonuses = record.meta?.bonuses ?? []
        let penalties = record.meta?.penalties ?? []

        Text("EDGAR Ex-99")
            .appStyle(.cardTitle)

        let justificationPairs = edgarJustificationParagraphs(ex?.justifications)
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

        let topRows = edgarTopLevelEx99Rows(ex)
        if !topRows.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(topRows.enumerated()), id: \.offset) { _, row in
                    edgarKeyValueRow(label: row.0, value: row.1)
                }
            }
            .padding(.top, justificationPairs.isEmpty ? 0 : 6)
        }

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

        geminiHealthBlock()
    }

    private func edgarJustificationParagraphs(_ j: HealthEx99Justifications?) -> [(String, String)] {
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

    private func edgarTopLevelEx99Rows(_ ex: HealthEx99Payload?) -> [(String, String)] {
        guard let ex else { return [] }
        let pairs: [(String, String?)] = [
            ("Expectation", ex.expectation),
            ("Guidance", ex.guidance),
            ("Market Reaction", ex.marketReaction),
            ("Past Performance", ex.pastPerformance),
        ]
        return pairs.compactMap { label, v in
            guard let t = v?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
            return (label, t)
        }
    }

    @ViewBuilder
    private func edgarKeyValueRow(label: String, value: String?) -> some View {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.isEmpty { EmptyView() } else {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(label):")
                    .appStyle(.detailFieldLabel)
                Text(trimmed)
                    .detailBody()
            }
        }
    }

    @ViewBuilder
    private func edgarBulletList(title: String, items: [String], maxItems: Int) -> some View {
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

    @ViewBuilder
    private func advisorHealthSections() -> some View {
        healthMetricRow(label: "Confidence", value: record.confidenceScore?.display)
        healthMetricRow(label: "Health", value: record.healthScore?.display)
        healthMetricRow(label: "Valuation", value: record.valuationScore?.display)
        healthMetricRow(label: "Piotroski", value: piotroskiDisplay(record.piotroski))
        healthMetricRow(label: "Altman Z", value: record.altmanZ?.display)

        if record.overlayPoints != nil || !record.overlayReasons.isEmpty {
            healthMetricRow(label: "Points", value: overlayPointsLabel(record.overlayPoints))
                .padding(.top, 2)
            if !record.overlayReasons.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(record.overlayReasons.enumerated()), id: \.offset) { _, reason in
                        Text("• \(reason)")
                            .appStyle(.detailBodyMuted)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }

        geminiHealthBlock()
    }

    @ViewBuilder
    private func geminiHealthBlock() -> some View {
        if record.geminiWeight != nil || record.geminiRec != nil || (record.geminiExplanation.map { $0.display != "—" } ?? false) {
            healthMetricRow(label: "Weight", value: record.geminiWeight?.display)
                .padding(.top, 2)
            healthMetricRow(label: "Recommendation", value: record.geminiRec?.display)
            if let gem = record.geminiExplanation?.display, gem != "—" {
                Text(gem)
                    .detailBody()
            }
        }
    }

    private func healthMetricRow(label: String, value: String?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .appStyle(.detailRowLabel)
            Spacer()
            Text(value ?? "—")
                .appStyle(.detailRowValue)
                .multilineTextAlignment(.trailing)
        }
    }

    private func piotroskiDisplay(_ scalar: HealthScalar?) -> String? {
        guard let s = scalar?.display, s != "—" else { return nil }
        if s.contains("/") { return s }
        if Int(s) != nil { return "\(s)/4" }
        return s
    }

    private func overlayPointsLabel(_ pts: Double?) -> String? {
        guard let pts else { return nil }
        if pts >= 0 {
            return String(format: "+%.1f", pts)
        }
        return String(format: "%.1f", pts)
    }

    private func formatHealthDate(_ iso: String?) -> String {
        guard let iso, !iso.isEmpty else { return "—" }
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let date = withFraction.date(from: iso) ?? plain.date(from: iso)
        guard let date else {
            return iso
        }
        let out = DateFormatter()
        out.dateStyle = .medium
        out.timeStyle = .short
        return out.string(from: date)
    }
}

import SwiftUI

/// Structured EDGAR discovery card (Discovery.meta from API).
struct EdgarDiscoveryMetaCard: View {
    let meta: DiscoveryMetaPayload

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(meta.lead?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (meta.lead ?? "8-K earnings filing")
                : "8-K earnings filing")
                .appStyle(.cardTitle)

            EdgarStructuredSections.edgarKeyValueRow(label: "Accession", value: meta.accession)
            if let weight = meta.weight {
                EdgarStructuredSections.edgarKeyValueRow(label: "Weight", value: String(format: "%.2f", weight))
            }
            ForEach(Array(EdgarStructuredSections.edgarTopLevelEx99Rows(meta.ex99).enumerated()), id: \.offset) { _, row in
                EdgarStructuredSections.edgarKeyValueRow(label: row.0, value: row.1)
            }
            if let open = meta.open, open.price != nil {
                let vs = open.vsClosePct.map { String(format: " (%.1f%% vs close)", $0) } ?? ""
                let priceText = open.price.map { String(format: "%.2f%@", $0, vs) }
                EdgarStructuredSections.edgarKeyValueRow(label: "Open", value: priceText)
            }

            Text("EX-99")
                .appStyle(.cardTitle)
                .padding(.top, 6)
            EdgarStructuredSections.justificationsBody(meta.ex99)
            EdgarStructuredSections.mediaBody(meta.media)
            EdgarStructuredSections.bonusesBody(
                bonuses: meta.bonuses ?? [],
                penalties: meta.penalties ?? []
            )

            if let urlText = meta.secUrl?.trimmingCharacters(in: .whitespacesAndNewlines), !urlText.isEmpty,
               let url = URL(string: urlText) {
                Link(urlText, destination: url)
                    .appStyle(.detailBodyMuted)
                    .padding(.top, 4)
            }
        }
    }
}

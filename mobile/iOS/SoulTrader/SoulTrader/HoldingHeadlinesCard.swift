import SwiftUI

/// Yahoo news titles on holding detail (parity with web `renderHeadlinesSection`).
struct HoldingHeadlinesCard: View {
    let headlines: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent headlines (7 days)")
                .appStyle(.cardTitle)

            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(headlines.enumerated()), id: \.offset) { _, title in
                    HStack(alignment: .top, spacing: 8) {
                        Text("•")
                            .appStyle(.bodyExplanation)
                            .foregroundStyle(Theme.secondaryText)
                            .fixedSize(horizontal: true, vertical: false)
                        Text(title)
                            .detailBody()
                    }
                }
            }

            Text("Yahoo Finance · titles only · not investment advice")
                .appStyle(.sectionCaption)
                .foregroundStyle(Theme.secondaryText)
        }
        .cardSurface()
    }
}

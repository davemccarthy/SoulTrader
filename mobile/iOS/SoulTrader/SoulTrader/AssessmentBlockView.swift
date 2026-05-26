import SwiftUI

/// v2 assessment panel — parity with `core/templates/core/includes/assessment_block.html`.
struct AssessmentBlockView: View {
    let scoring: DiscoveryScoring

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Assessment")
                    .appStyle(.cardTitle)
                Spacer()
                Text(scoring.displayScoreText)
                    .appStyle(.metricValue)
            }

            if let summary = scoring.summary {
                summarySection(summary)
            }

            if !scoring.components.isEmpty {
                componentsSection(scoring.components)
            }
        }
        .cardSurface()
    }

    private func summarySection(_ summary: DiscoveryScoringSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionCaption("Summary")
            assessmentRow(
                label: "Score",
                value: DiscoveryScoring.formatOptionalScore(summary.assessmentScore)
            )
            assessmentRow(
                label: "Weight",
                value: summary.discoveryWeightDisplay?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                    ? (summary.discoveryWeightDisplay ?? "—")
                    : "—"
            )
            assessmentRow(
                label: "Grade",
                value: summary.grade?.displayLetter ?? "—"
            )
            if let adjusted = summary.adjustedGrade {
                assessmentRow(label: "Adjusted", value: adjusted.displayLetter)
            }
        }
        .padding(.top, 4)
    }

    private func componentsSection(_ components: [DiscoveryScoringComponent]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionCaption("Details")
            ForEach(components) { comp in
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(comp.label)
                        .appStyle(.detailRowLabel)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("\(comp.weightPercent)%")
                        .appStyle(.detailRowValueAccent)
                        .frame(width: 40, alignment: .trailing)
                    Text(DiscoveryScoring.formatOptionalScore(comp.score))
                        .appStyle(.detailRowValue)
                        .frame(width: 48, alignment: .trailing)
                }
            }
        }
        .padding(.top, 6)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.white.opacity(0.12))
                .frame(height: 1)
                .offset(y: -3)
        }
    }

    private func sectionCaption(_ title: String) -> some View {
        Text(title)
            .appStyle(.sectionCaption)
    }

    private func assessmentRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .appStyle(.detailRowLabel)
            Spacer(minLength: 8)
            Text(value)
                .appStyle(.detailRowValue)
        }
    }
}

/// v2 assessment + optional v1 health cards (discovery detail and holding detail).
struct AssessmentAndHealthSectionView: View {
    let scoring: DiscoveryScoring?
    let healthRecords: [HealthHistoryRecord]
    var emptyMessage: String = "No assessment recorded."

    var body: some View {
        let showAssessment = scoring?.isV2 == true
        let showHealth = !healthRecords.isEmpty

        Group {
            if showAssessment || showHealth {
                VStack(spacing: 10) {
                    if let scoring, scoring.isV2 {
                        AssessmentBlockView(scoring: scoring)
                    }
                    ForEach(Array(healthRecords.enumerated()), id: \.element.id) { index, record in
                        HealthHistoryRecordCard(
                            record: record,
                            checkNumber: healthRecords.count > 1 ? index + 1 : nil
                        )
                    }
                }
            } else {
                Text(emptyMessage)
                    .appStyle(.emptyStateMessage)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .cardSurface()
            }
        }
    }
}

import SwiftUI

/// v2 assessment panel — parity with `core/templates/core/includes/assessment_block.html`.
struct AssessmentBlockView: View {
    let scoring: DiscoveryScoring

    private static let riskBandOrder: [(key: String, label: String)] = [
        ("CONSERVATIVE", "Conservative"),
        ("MODERATE", "Moderate"),
        ("AGGRESSIVE", "Aggressive"),
        ("RECKLESS", "Reckless"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            headerRow

            axesSection
                .padding(.top, 10)

            if !scoring.components.isEmpty {
                detailsSection
                    .padding(.top, 10)
            }

            if scoring.riskMatrix != nil, scoring.riskFloors != nil {
                riskBandSection
                    .padding(.top, 10)
            }

            if let interpretation = scoring.interpretation?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !interpretation.isEmpty {
                Text(interpretation)
                    .appStyle(.detailBodyMuted)
                    .padding(.top, 8)
            }
        }
        .cardSurface()
    }

    private var headerRow: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Assessment")
                .appStyle(.cardTitle)
            Spacer(minLength: 8)
            if let grade = scoring.displayGradeText, grade != "—" {
                VStack(alignment: .trailing, spacing: 1) {
                    Text("Grade")
                        .appStyle(.metricLabel)
                    Text(grade)
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .kerning(1.2)
                        .foregroundStyle(Theme.valuePrimary)
                }
            } else {
                Text(scoring.displayScoreText)
                    .appStyle(.metricValue)
            }
        }
    }

    private var axesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            assessmentTableHeader(columns: ("Axis", "Grade", "Score"))
            axisRow(
                label: "Stability",
                grade: scoring.stabilityGrade?.displayLetter,
                score: scoring.stability
            )
            axisRow(
                label: "Opportunity",
                grade: scoring.opportunityGrade?.displayLetter,
                score: scoring.opportunity
            )
            if scoring.showsOpportunityUpgrade {
                axisRow(
                    label: "Upgrade",
                    grade: scoring.opportunityAdjustedGrade?.displayLetter,
                    scoreText: scoring.opportunityUpgradeDisplayText
                )
            }
        }
    }

    private var detailsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionDivider
            assessmentTableHeader(columns: ("Details", "Wt", "Score"))
            ForEach(scoring.components) { comp in
                assessmentDataRow(
                    label: comp.label,
                    mid: "\(comp.weightPercent)%",
                    trailing: DiscoveryScoring.formatOptionalScore(comp.score),
                    midAccent: true
                )
            }
        }
    }

    private var riskBandSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionDivider
            assessmentTableHeader(columns: ("Risk band", "Floors", "Fit"))
            ForEach(Self.riskBandOrder, id: \.key) { band in
                let fit = scoring.riskMatrix?[band.key] ?? "—"
                let floor = scoring.riskFloors?[band.key]?.soCompositeFloor
                    ?? scoring.riskFloors?[band.key]?.soFloorDisplay
                    ?? "—"
                assessmentDataRow(
                    label: band.label,
                    mid: floor,
                    trailing: fit,
                    trailingColor: fitColor(for: fit)
                )
            }
        }
    }

    private var sectionDivider: some View {
        Rectangle()
            .fill(Color.white.opacity(0.12))
            .frame(height: 1)
            .padding(.bottom, 2)
    }

    private func axisRow(label: String, grade: String?, score: Double?) -> some View {
        assessmentDataRow(
            label: label,
            mid: grade ?? "—",
            trailing: DiscoveryScoring.formatOptionalScore(score),
            midAccent: true
        )
    }

    private func axisRow(label: String, grade: String?, scoreText: String) -> some View {
        assessmentDataRow(
            label: label,
            mid: grade ?? "—",
            trailing: scoreText,
            midAccent: true
        )
    }

    private func assessmentTableHeader(columns: (String, String, String)) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(columns.0)
                .appStyle(.sectionCaption)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(columns.1)
                .appStyle(.sectionCaption)
                .frame(width: 44, alignment: .center)
            Text(columns.2)
                .appStyle(.sectionCaption)
                .frame(width: 72, alignment: .trailing)
        }
    }

    private func assessmentDataRow(
        label: String,
        mid: String,
        trailing: String,
        midAccent: Bool = false,
        trailingColor: Color? = nil
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Text(label)
                .appStyle(.detailRowLabel)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(mid)
                .appStyle(midAccent ? .detailRowValueAccent : .detailRowValue)
                .kerning(midAccent ? 0.8 : 0)
                .frame(width: 44, alignment: .center)
            Text(trailing)
                .appStyle(.detailRowValue, color: trailingColor)
                .frame(width: 72, alignment: .trailing)
        }
    }

    private func fitColor(for fit: String) -> Color? {
        switch fit.uppercased() {
        case "BUY": return Theme.positive
        case "AVOID": return Theme.negative
        default: return nil
        }
    }
}

/// v2 assessment + optional v1 health cards (discovery detail and holding detail).
struct AssessmentAndHealthSectionView: View {
    let scoring: DiscoveryScoring?
    let healthRecords: [HealthHistoryRecord]
    var headlines: [String] = []
    var emptyMessage: String = "No assessment recorded."

    var body: some View {
        let showAssessment = scoring?.isV2 == true
        let showHealth = !healthRecords.isEmpty
        let showHeadlines = !headlines.isEmpty

        Group {
            if showAssessment || showHealth || showHeadlines {
                VStack(spacing: 10) {
                    if let scoring, scoring.isV2 {
                        AssessmentBlockView(scoring: scoring)
                    }
                    if showHeadlines {
                        HoldingHeadlinesCard(headlines: headlines)
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

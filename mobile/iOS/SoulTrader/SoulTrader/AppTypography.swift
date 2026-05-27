import SwiftUI

/// Semantic text styles — single source of truth for fonts, weights, and default colors.
enum AppTextStyle {
    /// Uppercase metric label (CURRENT, SCORE, PORTFOLIO).
    case metricLabel
    /// Metric value under a label.
    case metricValue
    /// Inline label prefix (e.g. "ABV:").
    case inlineMetricLabel
    /// Inline value beside a prefix.
    case inlineMetricValue
    /// Card or section title on dark surfaces.
    case cardTitle
    /// Primary line in list rows (company, advisor name).
    case listHeadline
    /// Secondary line in list rows (explanation, subtitle).
    case listSubline
    /// Detail screen title (symbol · company).
    case screenHeadline
    /// Detail screen subtitle (date, industry).
    case screenSubline
    /// Long-form prose in detail cards (discovery explanation, health narrative, Gemini).
    case bodyExplanation
    /// Secondary prose (bullets, empty hints inside detail cards).
    case detailBodyMuted
    /// Date or meta line under a detail card title.
    case detailMeta
    /// Inline field label before prose ("Past Performance:", health metric name).
    case detailFieldLabel
    /// Ticker under a logo.
    case tickerSymbol
    /// Empty state primary message.
    case emptyStateTitle
    /// Empty state secondary hint.
    case emptyStateMessage
    /// Section caption inside cards (Summary, Details).
    case sectionCaption
    /// Label in a detail/assessment row.
    case detailRowLabel
    /// Value in a detail/assessment row.
    case detailRowValue
    /// Emphasized value in a detail row (e.g. weight %).
    case detailRowValueAccent

    var font: Font {
        switch self {
        case .metricLabel, .inlineMetricLabel:
            return .caption2
        case .metricValue, .cardTitle, .listHeadline, .emptyStateMessage:
            return .subheadline
        case .inlineMetricValue:
            return .caption
        case .listSubline, .screenSubline, .sectionCaption:
            return .caption
        case .screenHeadline, .emptyStateTitle:
            return .headline
        case .bodyExplanation, .detailBodyMuted, .detailMeta,
             .detailRowLabel, .detailRowValue, .detailRowValueAccent, .detailFieldLabel:
            return .subheadline
        case .tickerSymbol:
            return .system(size: 11, weight: .bold)
        }
    }

    var weight: Font.Weight? {
        switch self {
        case .metricLabel, .metricValue, .inlineMetricValue, .cardTitle,
             .listHeadline, .screenSubline, .sectionCaption, .detailRowLabel, .detailRowValue,
             .detailRowValueAccent, .detailFieldLabel, .emptyStateMessage:
            return .semibold
        case .screenHeadline, .emptyStateTitle:
            return .bold
        case .bodyExplanation, .detailBodyMuted, .detailMeta:
            return .light
        case .inlineMetricLabel, .listSubline, .tickerSymbol:
            return nil
        }
    }

    var color: Color {
        switch self {
        case .metricLabel, .cardTitle, .sectionCaption, .detailRowValueAccent, .inlineMetricLabel,
             .detailRowLabel, .detailFieldLabel:
            return Theme.labelAccent
        case .metricValue, .listHeadline, .inlineMetricValue, .detailRowValue, .tickerSymbol:
            return Theme.valuePrimary
        case .listSubline, .screenSubline, .emptyStateMessage, .detailBodyMuted, .detailMeta:
            return Theme.secondaryText
        case .screenHeadline, .emptyStateTitle:
            return .white
        case .bodyExplanation:
            return Theme.valuePrimary.opacity(0.95)
        }
    }
}

extension Text {
    func appStyle(_ style: AppTextStyle, color: Color? = nil) -> some View {
        var text = self.font(style.font)
        if let weight = style.weight {
            text = text.fontWeight(weight)
        }
        return text.foregroundStyle(color ?? style.color)
    }

    /// Multiline detail prose (discovery explanation, health narrative).
    func detailBody() -> some View {
        appStyle(.bodyExplanation)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}

extension View {
    @ViewBuilder
    func appStyle(_ style: AppTextStyle, color: Color? = nil) -> some View {
        if let weight = style.weight {
            font(style.font)
                .fontWeight(weight)
                .foregroundStyle(color ?? style.color)
        } else {
            font(style.font)
                .foregroundStyle(color ?? style.color)
        }
    }
}

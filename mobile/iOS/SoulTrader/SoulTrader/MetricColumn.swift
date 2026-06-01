import SwiftUI

/// Label + value column used in summary strips and detail metric rows.
struct MetricColumn: View {
    let title: String
    let value: String
    var valueColor: Color = Theme.valuePrimary
    var alignment: HorizontalAlignment = .leading
    var expands: Bool = false

    var body: some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: Theme.metricSpacing) {
            Text(title)
                .appStyle(.metricLabel)
            Text(value)
                .appStyle(.metricValue, color: valueColor)
                .lineLimit(1)
        }
        .frame(maxWidth: expands ? .infinity : nil, alignment: frameAlignment)
    }

    private var frameAlignment: Alignment {
        alignment == .leading ? .leading : .trailing
    }
}

/// Compact inline label:value (e.g. "APR: +1.2%").
struct InlineMetricPair: View {
    let title: String
    let value: String
    var valueColor: Color = Theme.valuePrimary

    var body: some View {
        HStack(spacing: 4) {
            Text(title)
                .appStyle(.inlineMetricLabel)
            Text(value)
                .appStyle(.inlineMetricValue, color: valueColor)
        }
    }
}

extension View {
    /// Standard dark card surface used across list and detail screens.
    func cardSurface(
        vertical: CGFloat = 8,
        horizontal: CGFloat = 10
    ) -> some View {
        padding(.vertical, vertical)
            .padding(.horizontal, horizontal)
            .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius))
    }
}

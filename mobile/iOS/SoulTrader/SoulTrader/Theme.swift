import SwiftUI

enum Theme {
    // Layout
    static let cardCornerRadius: CGFloat = 10
    static let metricSpacing: CGFloat = 2

    // Base surfaces
    static let appBackground = LinearGradient(
        stops: [
            .init(color: Color(red: 0.00, green: 0.42, blue: 0.20), location: 0.00),
            .init(color: Color(red: 0.00, green: 0.24, blue: 0.12), location: 0.24),
            .init(color: Color(red: 0.00, green: 0.10, blue: 0.06), location: 1.00),
        ],
        startPoint: .bottomTrailing,
        endPoint: .topLeading
    )
    static let rowBackground = Color(red: 0.18, green: 0.18, blue: 0.20).opacity(0.76)

    // Brand accents
    static let brandHeaderStart = Color(red: 0.0, green: 0.52, blue: 0.24)
    static let brandHeaderEnd = Color(red: 0.0, green: 0.69, blue: 0.31)
    static let brandSubtitle = Color(red: 0.98, green: 0.81, blue: 0.20)

    // Content colors for dark rows
    static let labelAccent = Color(red: 0.96, green: 0.84, blue: 0.50)
    static let valuePrimary = Color(red: 0.96, green: 0.96, blue: 0.96)
    static let secondaryText = Color.white.opacity(0.75)
    static let positive = Color.green
    static let negative = Color.red
    static let link = Color(red: 0.45, green: 0.78, blue: 1.0)

    /// Color for signed percent or currency deltas (green / red / neutral).
    static func signedColor(for value: Double?) -> Color {
        guard let value else { return valuePrimary }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        if normalized > 0 { return positive }
        if normalized < 0 { return negative }
        return valuePrimary
    }

    static func signedColor(for value: Decimal?) -> Color {
        guard let value else { return valuePrimary }
        if value > 0 { return positive }
        if value < 0 { return negative }
        return valuePrimary
    }

    /// Summary-strip currency: `$2.3M`, `$850K`, `$999` (preserves sign).
    static func formatCompactCurrency(_ value: Double) -> String {
        let absValue = abs(value)
        let sign = value < 0 ? "-" : ""
        if absValue >= 1_000_000 {
            return String(format: "%@$%.1fM", sign, absValue / 1_000_000)
        }
        if absValue >= 1_000 {
            return String(format: "%@$%.0fK", sign, absValue / 1_000)
        }
        return String(format: "%@$%.0f", sign, absValue)
    }
}

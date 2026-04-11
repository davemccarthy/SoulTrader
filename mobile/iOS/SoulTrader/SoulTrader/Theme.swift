import SwiftUI

enum Theme {
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
}

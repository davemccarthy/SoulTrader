import SwiftUI

enum Theme {
    // Base surfaces
    static let appBackground = Color.black
    static let rowBackground = Color.black

    // Brand accents
    static let brandHeaderStart = Color(red: 0.0, green: 0.52, blue: 0.24)
    static let brandHeaderEnd = Color(red: 0.0, green: 0.69, blue: 0.31)
    static let brandSubtitle = Color(red: 0.98, green: 0.81, blue: 0.20)

    // Content colors for dark rows
    static let labelAccent = Color(red: 0.96, green: 0.84, blue: 0.50)
    static let valuePrimary = Color(red: 0.96, green: 0.96, blue: 0.96)
    static let secondaryText = Color.white.opacity(0.75)
}

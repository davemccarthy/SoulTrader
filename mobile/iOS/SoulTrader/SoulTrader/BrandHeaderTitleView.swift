import SwiftUI

struct BrandHeaderTitleView: View {
    let title: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.headline)
                .fontWeight(.bold)
                .foregroundStyle(.white)
            Text("KLYNT INDUSTRIES")
                .font(.caption2)
                .fontWeight(.black)
                .foregroundStyle(Theme.brandSubtitle)
        }
    }
}

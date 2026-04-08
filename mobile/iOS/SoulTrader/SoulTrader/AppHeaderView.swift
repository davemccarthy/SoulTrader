import SwiftUI

struct AppHeaderView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.headerTitle)
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                Text("KLYNT INDUSTRIES")
                    .font(.caption2)
                    .fontWeight(.black)
                    .foregroundStyle(Theme.brandSubtitle)
            }
            Spacer()
            Button {
                Task { await viewModel.refreshAll() }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .foregroundStyle(.white)
            }
            .padding(.trailing, 10)
            Button("Logout", role: .destructive) {
                viewModel.logout()
            }
            .font(.subheadline)
            .foregroundStyle(.white)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity)
        .background(
            LinearGradient(
                colors: [Theme.brandHeaderStart, Theme.brandHeaderEnd],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
    }
}

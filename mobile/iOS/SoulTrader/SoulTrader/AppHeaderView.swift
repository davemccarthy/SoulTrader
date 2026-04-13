import SwiftUI

struct AppHeaderView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        HStack {
            BrandHeaderTitleView(title: viewModel.headerTitle)
            Spacer()
            Button {
                Task { await viewModel.refreshAll() }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .foregroundStyle(.white)
            }
            .padding(.trailing, 10)
            Button(viewModel.hasSelectedFund ? "Funds" : "Logout", role: viewModel.hasSelectedFund ? nil : .destructive) {
                if viewModel.hasSelectedFund {
                    viewModel.selectedTab = .funds
                    viewModel.clearSelectedFund()
                } else {
                    viewModel.logout()
                }
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

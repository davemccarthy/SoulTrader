import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AuthViewModel()

    init() {
        configureTabBarAppearance()
    }

    private var guardedTabSelection: Binding<AppTab> {
        Binding(
            get: { viewModel.selectedTab },
            set: { newTab in
                if newTab == .funds {
                    viewModel.selectedTab = .funds
                    viewModel.clearSelectedFund()
                    return
                }
                guard viewModel.hasSelectedFund else {
                    viewModel.selectedTab = .funds
                    return
                }
                viewModel.selectedTab = newTab
            }
        )
    }

    var body: some View {
        ZStack {
            Group {
                if viewModel.isAuthenticated {
                    VStack(spacing: 0) {
                        AppHeaderView(viewModel: viewModel)
                        TabView(selection: guardedTabSelection) {
                            NavigationStack { FundsView(viewModel: viewModel) }
                                .background(appBackground)
                                .tabItem { Label("FUNDS", systemImage: "dollarsign.circle") }
                                .tag(AppTab.funds)

                            HoldingsView(viewModel: viewModel)
                                .background(appBackground)
                                .tabItem { Label("HOLDINGS", systemImage: "chart.pie") }
                                .tag(AppTab.holdings)
                                .disabled(viewModel.selectedTab == .funds || !viewModel.hasSelectedFund)

                            NavigationStack { TradesView(viewModel: viewModel) }
                                .background(appBackground)
                                .tabItem { Label("TRADES", systemImage: "arrow.left.arrow.right") }
                                .tag(AppTab.trades)
                                .disabled(viewModel.selectedTab == .funds || !viewModel.hasSelectedFund)
                        }
                        .background(appBackground)
                        .onChange(of: viewModel.selectedTab) { _, newTab in
                            if newTab == .funds { viewModel.clearSelectedFund() }
                        }
                        .onChange(of: viewModel.selectedFundId) { _, newFundId in
                            if newFundId == nil && viewModel.selectedTab != .funds {
                                viewModel.selectedTab = .funds
                            }
                        }
                    }
                    .background(appBackground)
                } else {
                    LoginView(viewModel: viewModel)
                }
            }
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView("Working...")
                    .tint(.white)
                    .foregroundStyle(.white)
                    .padding()
                    .background(Color.black.opacity(0.72), in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .task {
            await viewModel.bootstrap()
        }
    }
}

#Preview {
    ContentView()
}

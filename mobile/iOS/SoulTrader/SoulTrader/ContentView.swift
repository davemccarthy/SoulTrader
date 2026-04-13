import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AuthViewModel()

    init() {
        configureTabBarAppearance()
    }

    private var fundSessionTabSelection: Binding<AppTab> {
        Binding(
            get: {
                if viewModel.selectedTab == .trades { return .trades }
                return .holdings
            },
            set: { newTab in
                guard viewModel.hasSelectedFund else {
                    viewModel.selectedTab = .funds
                    return
                }
                viewModel.selectedTab = (newTab == .trades) ? .trades : .holdings
            }
        )
    }

    var body: some View {
        ZStack {
            Group {
                if viewModel.isAuthenticated {
                    VStack(spacing: 0) {
                        AppHeaderView(viewModel: viewModel)
                        if viewModel.hasSelectedFund {
                            TabView(selection: fundSessionTabSelection) {
                                HoldingsView(viewModel: viewModel)
                                    .background(appBackground)
                                    .tabItem { Label("HOLDINGS", systemImage: "chart.pie") }
                                    .tag(AppTab.holdings)

                                TradesView(viewModel: viewModel)
                                    .background(appBackground)
                                    .tabItem { Label("TRADES", systemImage: "arrow.left.arrow.right") }
                                    .tag(AppTab.trades)
                            }
                            .background(appBackground)
                        } else {
                            NavigationStack {
                                FundsView(viewModel: viewModel)
                            }
                            .background(appBackground)
                        }
                    }
                    .background(appBackground)
                    .onChange(of: viewModel.selectedFundId) { _, newFundId in
                        if newFundId == nil {
                            viewModel.selectedTab = .funds
                        } else if viewModel.selectedTab == .funds {
                            viewModel.selectedTab = .holdings
                        }
                    }
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

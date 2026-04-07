//
//  ContentView.swift
//  SoulTrader
//
//  Created by David McCarthy on 07/04/2026.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AuthViewModel()

    var body: some View {
        Group {
            if viewModel.isAuthenticated {
                VStack(spacing: 0) {
                    AppHeaderView(viewModel: viewModel)
                    TabView(selection: $viewModel.selectedTab) {
                        NavigationStack {
                            FundsView(viewModel: viewModel)
                        }
                        .tabItem { Label("FUNDS", systemImage: "dollarsign.circle") }
                        .tag(AppTab.funds)

                        NavigationStack {
                            HoldingsView(viewModel: viewModel)
                        }
                        .tabItem { Label("HOLDINGS", systemImage: "chart.pie") }
                        .tag(AppTab.holdings)

                        NavigationStack {
                            TradesView(viewModel: viewModel)
                        }
                        .tabItem { Label("TRADES", systemImage: "arrow.left.arrow.right") }
                        .tag(AppTab.trades)
                    }
                }
            } else {
                LoginView(viewModel: viewModel)
            }
        }
        .overlay {
            if viewModel.isLoading {
                ProgressView("Working...")
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .task {
            await viewModel.bootstrap()
        }
    }
}

private struct AppHeaderView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("SOULTRADER")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                Text("KLYNT INDUSTRIES")
                    .font(.caption2)
                    .fontWeight(.black)
                    .foregroundStyle(Color(red: 0.98, green: 0.81, blue: 0.20))
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
                colors: [Color(red: 0.0, green: 0.52, blue: 0.24), Color(red: 0.0, green: 0.69, blue: 0.31)],
                startPoint: .leading,
                endPoint: .trailing
            )
        )
    }
}

private enum AppTab {
    case funds
    case holdings
    case trades
}

private struct UserProfile: Decodable {
    let id: Int
    let username: String
    let email: String
}

private struct TokenResponse: Decodable {
    let access: String
    let refresh: String
}

private struct ProfileResponse: Decodable {
    let user: UserProfile
    let risk: String
    let investment: String
    let cash: String
}

private struct FundDashboardResponse: Decodable {
    let totalValue: Double
    let returnAmount: Double
    let cash: Double
    let holdingsCount: Int
    let holdingsPnl: Double
    let returnPercent: Double
    let estAbvPercent: Double
    let estabDays: Int

    private enum CodingKeys: String, CodingKey {
        case totalValue = "total_value"
        case returnAmount = "return_amount"
        case cash
        case holdingsCount = "holdings_count"
        case holdingsPnl = "holdings_pnl"
        case returnPercent = "return_percent"
        case estAbvPercent = "est_abv_percent"
        case estabDays = "estab_days"
    }
}

private struct FundResponse: Decodable, Identifiable {
    let id: Int
    let name: String
    let spread: String?
    let risk: String
    let advisors: [String]
    let dashboard: FundDashboardResponse
}

private struct StockInfo: Decodable {
    let symbol: String
    let company: String?
    let price: String?
}

private struct HoldingResponse: Decodable, Identifiable {
    let id: Int
    let stock: StockInfo
    let shares: String
    let averagePrice: String

    private enum CodingKeys: String, CodingKey {
        case id
        case stock
        case shares
        case averagePrice = "average_price"
    }
}

private struct TradeResponse: Decodable, Identifiable {
    let id: Int
    let stock: StockInfo
    let action: String
    let price: String
    let shares: String
    let sa: String?
}

private struct LoginRequest: Encodable {
    let username: String
    let password: String
}

private struct APIEnvironment {
    // For iOS Simulator use localhost; for physical device use your machine IP.
    static let baseURL = URL(string: "http://127.0.0.1:8000/api/")!
}

private struct TokenStore {
    private enum Keys {
        static let access = "auth.accessToken"
        static let refresh = "auth.refreshToken"
    }

    func getAccessToken() -> String? {
        UserDefaults.standard.string(forKey: Keys.access)
    }

    func save(access: String, refresh: String) {
        UserDefaults.standard.set(access, forKey: Keys.access)
        UserDefaults.standard.set(refresh, forKey: Keys.refresh)
    }

    func clear() {
        UserDefaults.standard.removeObject(forKey: Keys.access)
        UserDefaults.standard.removeObject(forKey: Keys.refresh)
    }
}

private struct APIClient {
    let baseURL: URL
    let session: URLSession = .shared

    private func endpoint(_ relativePath: String) -> URL {
        URL(string: relativePath, relativeTo: baseURL)!.absoluteURL
    }

    func login(username: String, password: String) async throws -> TokenResponse {
        let url = endpoint("auth/login/")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(LoginRequest(username: username, password: password))

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(TokenResponse.self, from: data)
    }

    func fetchCurrentUser(accessToken: String) async throws -> UserProfile {
        let url = endpoint("auth/user/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(UserProfile.self, from: data)
    }

    func fetchFunds(accessToken: String) async throws -> [FundResponse] {
        let url = endpoint("funds/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([FundResponse].self, from: data)
    }

    func fetchHoldings(accessToken: String) async throws -> [HoldingResponse] {
        let url = endpoint("holdings/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([HoldingResponse].self, from: data)
    }

    func fetchTrades(accessToken: String) async throws -> [TradeResponse] {
        let url = endpoint("trades/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([TradeResponse].self, from: data)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.httpStatus(http.statusCode, body)
        }
    }
}

private enum APIError: Error, LocalizedError {
    case invalidResponse
    case missingToken
    case httpStatus(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid server response."
        case .missingToken:
            return "Missing access token."
        case let .httpStatus(code, body):
            if body.isEmpty { return "Request failed (\(code))." }
            return "Request failed (\(code)): \(body)"
        }
    }
}

@MainActor
private final class AuthViewModel: ObservableObject {
    @Published var username = ""
    @Published var password = ""
    @Published var selectedTab: AppTab = .funds
    @Published var currentUser: UserProfile?
    @Published var funds: [FundResponse] = []
    @Published var holdings: [HoldingResponse] = []
    @Published var trades: [TradeResponse] = []
    @Published var isLoading = false
    @Published var statusMessage: String?

    private let tokenStore = TokenStore()
    private let apiClient = APIClient(baseURL: APIEnvironment.baseURL)

    var isAuthenticated: Bool {
        tokenStore.getAccessToken() != nil
    }

    var baseURLString: String {
        APIEnvironment.baseURL.absoluteString
    }

    func bootstrap() async {
        guard isAuthenticated else {
            statusMessage = "Not logged in."
            return
        }
        await refreshAll()
    }

    func login() async {
        isLoading = true
        defer { isLoading = false }

        do {
            let token = try await apiClient.login(username: username, password: password)
            tokenStore.save(access: token.access, refresh: token.refresh)
            password = ""
            selectedTab = .funds
            statusMessage = "Login successful."
            await refreshAll()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func refreshCurrentUser() async {
        isLoading = true
        defer { isLoading = false }

        do {
            guard let access = tokenStore.getAccessToken() else {
                throw APIError.missingToken
            }
            currentUser = try await apiClient.fetchCurrentUser(accessToken: access)
            statusMessage = "User loaded."
        } catch {
            currentUser = nil
            statusMessage = error.localizedDescription
        }
    }

    func refreshAll() async {
        isLoading = true
        defer { isLoading = false }

        do {
            guard let access = tokenStore.getAccessToken() else {
                throw APIError.missingToken
            }

            async let userTask = apiClient.fetchCurrentUser(accessToken: access)
            async let fundsTask = apiClient.fetchFunds(accessToken: access)
            async let holdingsTask = apiClient.fetchHoldings(accessToken: access)
            async let tradesTask = apiClient.fetchTrades(accessToken: access)

            currentUser = try await userTask
            funds = try await fundsTask
            holdings = try await holdingsTask
            trades = try await tradesTask
            statusMessage = "Data refreshed."
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func logout() {
        tokenStore.clear()
        currentUser = nil
        funds = []
        holdings = []
        trades = []
        username = ""
        password = ""
        selectedTab = .funds
        statusMessage = "Logged out."
    }
}

private struct LoginView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Login") {
                    TextField("Username", text: $viewModel.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Password", text: $viewModel.password)
                }

                Section {
                    Button("Login") {
                        Task { await viewModel.login() }
                    }
                    .disabled(viewModel.isLoading || viewModel.username.isEmpty || viewModel.password.isEmpty)
                }

                if let statusMessage = viewModel.statusMessage {
                    Section("Status") {
                        Text(statusMessage).font(.footnote)
                    }
                }
            }
            .navigationTitle("SoulTrader")
        }
    }
}

private struct FundsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        List(viewModel.funds) { fund in
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .top, spacing: 0) {
                    metricPair(
                        title: "FUND",
                        value: fund.name.isEmpty ? "Unnamed" : fund.name,
                        alignment: .leading,
                        isFlexible: false
                    )
                    .frame(width: 80, alignment: .leading)

                    metricPair(
                        title: "WEALTH",
                        value: formatCurrency(fund.dashboard.totalValue),
                        alignment: .leading
                    )

                    metricPair(
                        title: "PORTFOLIO",
                        value: formatPercent(fund.dashboard.holdingsPnl),
                        color: fund.dashboard.holdingsPnl >= 0 ? .green : .red,
                        alignment: .trailing
                    )

                    metricPair(
                        title: "P&L",
                        value: formatPercent(fund.dashboard.returnPercent),
                        color: fund.dashboard.returnPercent >= 0 ? .green : .red,
                        alignment: .trailing
                    )
                }

                HStack(spacing: 10) {
                    miniPair(
                        title: "DUR:",
                        value: "\(fund.dashboard.estabDays) days"
                    )
                    Spacer()
                    miniPair(
                        title: "EST ABV:",
                        value: formatPercent(fund.dashboard.estAbvPercent),
                        color: fund.dashboard.estAbvPercent >= 0 ? .green : .red
                    )
                    miniPair(
                        title: "TODAY:",
                        value: "0.00%"
                    )
                }
            }
            .padding(.vertical, 4)
            .listRowSeparator(.hidden)
        }
        .listStyle(.plain)
        .toolbar(.hidden, for: .navigationBar)
    }

    private func metricPair(
        title: String,
        value: String,
        color: Color = .primary,
        alignment: Alignment,
        isFlexible: Bool = true
    ) -> some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundStyle(color)
                .lineLimit(1)
        }
        .frame(maxWidth: isFlexible ? .infinity : nil, alignment: alignment)
    }

    private func miniPair(
        title: String,
        value: String,
        color: Color = .secondary
    ) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(color)
        }
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        //formatter.currencyCode = "USD"
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }

    private func formatPercent(_ value: Double) -> String {
        String(format: "%.2f%%", value)
    }

}

private struct HoldingsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        List(viewModel.holdings) { holding in
            VStack(alignment: .leading, spacing: 4) {
                Text(holding.stock.symbol).font(.headline)
                Text(holding.stock.company ?? "-").font(.subheadline).foregroundStyle(.secondary)
                Text("Shares: \(holding.shares)  Avg: \(holding.averagePrice)").font(.footnote)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
    }
}

private struct TradesView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        List(viewModel.trades) { trade in
            VStack(alignment: .leading, spacing: 4) {
                Text("\(trade.action) \(trade.stock.symbol)").font(.headline)
                Text("Shares: \(trade.shares)  Price: \(trade.price)").font(.footnote)
                if let sa = trade.sa, !sa.isEmpty {
                    Text("SA: \(sa)").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .toolbar(.hidden, for: .navigationBar)
    }
}

#Preview {
    ContentView()
}

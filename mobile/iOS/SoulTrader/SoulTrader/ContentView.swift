//
//  ContentView.swift
//  SoulTrader
//
//  Created by David McCarthy on 07/04/2026.
//

import SwiftUI
import UIKit

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
                            .disabled(viewModel.selectedTab == .funds || !viewModel.hasSelectedFund)

                            NavigationStack {
                                TradesView(viewModel: viewModel)
                            }
                            .tabItem { Label("TRADES", systemImage: "arrow.left.arrow.right") }
                            .tag(AppTab.trades)
                            .disabled(viewModel.selectedTab == .funds || !viewModel.hasSelectedFund)
                        }
                        .onChange(of: viewModel.selectedTab) { _, newTab in
                            if newTab == .funds {
                                viewModel.clearSelectedFund()
                            }
                        }
                        .onChange(of: viewModel.selectedFundId) { _, newFundId in
                            if newFundId == nil && viewModel.selectedTab != .funds {
                                viewModel.selectedTab = .funds
                            }
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

private func configureTabBarAppearance() {
    let appearance = UITabBarAppearance()
    appearance.configureWithOpaqueBackground()

    let gradientImage = tabBarGradientImage(
        size: CGSize(width: 600, height: 120),
        topLeft: UIColor(red: 0.10, green: 0.72, blue: 0.36, alpha: 1.0),
        bottomRight: UIColor(red: 0.00, green: 0.42, blue: 0.18, alpha: 1.0)
    )

    appearance.backgroundImage = gradientImage
    appearance.backgroundColor = .clear
    appearance.shadowColor = UIColor.black.withAlphaComponent(0.25)

    let normalColor = UIColor.white.withAlphaComponent(0.75)
    let selectedColor = UIColor.white

    appearance.stackedLayoutAppearance.normal.iconColor = normalColor
    appearance.stackedLayoutAppearance.normal.titleTextAttributes = [.foregroundColor: normalColor]
    appearance.stackedLayoutAppearance.selected.iconColor = selectedColor
    appearance.stackedLayoutAppearance.selected.titleTextAttributes = [.foregroundColor: selectedColor]

    UITabBar.appearance().standardAppearance = appearance
    UITabBar.appearance().scrollEdgeAppearance = appearance
}

private func tabBarGradientImage(
    size: CGSize,
    topLeft: UIColor,
    bottomRight: UIColor
) -> UIImage {
    let renderer = UIGraphicsImageRenderer(size: size)
    return renderer.image { context in
        let cgContext = context.cgContext
        let colors = [topLeft.cgColor, bottomRight.cgColor] as CFArray
        let locations: [CGFloat] = [0.0, 1.0]
        guard
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let gradient = CGGradient(colorsSpace: colorSpace, colors: colors, locations: locations)
        else { return }

        cgContext.drawLinearGradient(
            gradient,
            start: CGPoint(x: 0, y: 0),
            end: CGPoint(x: size.width, y: size.height),
            options: []
        )
    }
}

private struct AppHeaderView: View {
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
    let shares: Int
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
    let shares: Int
    let sa: Int?
}

private struct LoginRequest: Encodable {
    let username: String
    let password: String
}

private struct APIEnvironment {
    // For iOS Simulator use localhost; for physical device use your machine IP.
    static let baseURL = URL(string: "http://192.168.1.6:8000/api/")!
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

    func fetchHoldings(accessToken: String, fundId: Int?) async throws -> [HoldingResponse] {
        let base = endpoint("holdings/")
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        if let fundId {
            components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))]
        }
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([HoldingResponse].self, from: data)
    }

    func fetchTrades(accessToken: String, fundId: Int?) async throws -> [TradeResponse] {
        let base = endpoint("trades/")
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        if let fundId {
            components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))]
        }
        var request = URLRequest(url: components.url!)
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
    @Published var selectedFundId: Int?
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

    var hasSelectedFund: Bool {
        selectedFundId != nil
    }

    var baseURLString: String {
        APIEnvironment.baseURL.absoluteString
    }

    var selectedFundName: String? {
        guard let selectedFundId else { return nil }
        return funds.first(where: { $0.id == selectedFundId })?.name
    }

    var headerTitle: String {
        switch selectedTab {
        case .funds:
            return "SOULTRADER - FUNDS"
        case .holdings, .trades:
            if let selectedFundName, !selectedFundName.isEmpty {
                return "SOULTRADER - \(selectedFundName)"
            }
            return "SOULTRADER"
        }
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
            async let holdingsTask = apiClient.fetchHoldings(accessToken: access, fundId: selectedFundId)
            async let tradesTask = apiClient.fetchTrades(accessToken: access, fundId: selectedFundId)

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
        selectedFundId = nil
        funds = []
        holdings = []
        trades = []
        username = ""
        password = ""
        selectedTab = .funds
        statusMessage = "Logged out."
    }

    func selectFund(_ fundId: Int) async {
        selectedFundId = fundId
        selectedTab = .holdings
        await refreshAll()
    }

    func clearSelectedFund() {
        selectedFundId = nil
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
                    .frame(width: 64, alignment: .leading)

                    metricPair(
                        title: "WEALTH",
                        value: formatCurrency(fund.dashboard.totalValue),
                        alignment: .leading
                    )
                    .frame(width: 128, alignment: .leading)

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
            .contentShape(Rectangle())
            .onTapGesture {
                Task {
                    await viewModel.selectFund(fund.id)
                }
            }
            .background(
                viewModel.selectedFundId == fund.id
                    ? Color.green.opacity(0.08)
                    : Color.clear
            )
            .listRowBackground(Color.black)
            .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
        }
        .scrollContentBackground(.hidden)
        .scrollIndicators(.hidden)
        .background(Color.black)
        .toolbar(.hidden, for: .navigationBar)
    }

    private func metricPair(
        title: String,
        value: String,
        color: Color = Color(red: 0.96, green: 0.96, blue: 0.96),
        alignment: Alignment,
        isFlexible: Bool = true
    ) -> some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Color(red: 0.96, green: 0.84, blue: 0.50))
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
        color: Color = Color(red: 0.96, green: 0.96, blue: 0.96)
    ) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(Color(red: 0.96, green: 0.84, blue: 0.50))
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
        Group {
            if viewModel.holdings.isEmpty {
                VStack(spacing: 8) {
                    Text("No holdings to show.")
                        .font(.headline)
                        .foregroundStyle(.white)
                    Text("Select a fund with holdings on the FUNDS tab.")
                        .font(.footnote)
                        .foregroundStyle(Color.white.opacity(0.75))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .padding()
            } else {
                List(viewModel.holdings) { holding in
                    HStack(spacing: 12) {
                        imageTickerPair(symbol: holding.stock.symbol)
                        middleCompanyInvestmentPair(holding: holding)
                        Spacer()
                        pricePnlPair(holding: holding)
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, 6)
                    .background(Color.black, in: RoundedRectangle(cornerRadius: 10))
                    .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                    .listRowBackground(Color.clear)
                }
                .scrollContentBackground(.hidden)
                .scrollIndicators(.hidden)
                .background(Color.black)
            }
        }
        .background(Color.black)
        .toolbar(.hidden, for: .navigationBar)
    }

    private func imageTickerPair(symbol: String) -> some View {
        VStack(spacing: 4) {
            AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
                image
                    .resizable()
                    .scaledToFit()
            } placeholder: {
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.gray.opacity(0.15))
            }
            .frame(width: 34, height: 34)

            Text(symbol)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundStyle(.white)
                .lineLimit(1)
        }
        .frame(width: 56, alignment: .leading)
    }

    private func middleCompanyInvestmentPair(holding: HoldingResponse) -> some View {
        let avgDecimal = decimal(from: holding.averagePrice) ?? 0
        let sharesDecimal = Decimal(holding.shares)
        let investment = sharesDecimal * avgDecimal

        return VStack(alignment: .leading, spacing: 3) {
            Text(holding.stock.company ?? holding.stock.symbol)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Color(red: 0.96, green: 0.96, blue: 0.96))
                .lineLimit(1)
                .truncationMode(.tail)

            Text(formatCurrency(investment))
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(Color(red: 0.96, green: 0.96, blue: 0.96))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func pricePnlPair(holding: HoldingResponse) -> some View {
        let priceDecimal = decimal(from: holding.stock.price)
        let avgDecimal = decimal(from: holding.averagePrice)
        let pnlPercent = computePnlPercent(price: priceDecimal, average: avgDecimal)
        let pnlColor: Color = {
            guard let pnlPercent else { return Color(red: 0.96, green: 0.96, blue: 0.96) }
            if pnlPercent > 0 { return .green }
            if pnlPercent < 0 { return .red }
            return Color(red: 0.96, green: 0.96, blue: 0.96)
        }()

        return VStack(alignment: .trailing, spacing: 2) {
            Text(formatCurrency(priceDecimal))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Color(red: 0.96, green: 0.96, blue: 0.96))
                .lineLimit(1)

            Text(formatPercent(pnlPercent))
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(pnlColor)
                .lineLimit(1)
        }
        .frame(minWidth: 62, alignment: .trailing)
    }

    private func decimal(from text: String?) -> Decimal? {
        guard let text, !text.isEmpty else { return nil }
        return Decimal(string: text)
    }

    private func computePnlPercent(price: Decimal?, average: Decimal?) -> Double? {
        guard let price, let average, average != 0 else { return nil }
        let percent = ((price / average) - 1) * 100
        return NSDecimalNumber(decimal: percent).doubleValue
    }

    private func formatCurrency(_ value: Decimal?) -> String {
        guard let value else { return "$0.00" }
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSDecimalNumber(decimal: value)) ?? "$0.00"
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "0.00%" }
        return String(format: "%.2f%%", value)
    }
}

private struct TradesView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        List(viewModel.trades) { trade in
            VStack(alignment: .leading, spacing: 4) {
                Text("\(trade.action) \(trade.stock.symbol)").font(.headline)
                Text("Shares: \(trade.shares)  Price: \(trade.price)").font(.footnote)
                if let sa = trade.sa {
                    Text("SA: \(sa)").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 8)
            .background(Color.white, in: RoundedRectangle(cornerRadius: 10))
            .listRowInsets(EdgeInsets(top: 4, leading: 10, bottom: 4, trailing: 10))
            .listRowBackground(Color.clear)
        }
        .scrollContentBackground(.hidden)
        .scrollIndicators(.hidden)
        .background(Color.clear)
        .toolbar(.hidden, for: .navigationBar)
    }
}

#Preview {
    ContentView()
}

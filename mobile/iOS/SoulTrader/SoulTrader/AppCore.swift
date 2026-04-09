import SwiftUI
import UIKit
import Charts

let appBackground = Theme.appBackground

enum AppTab {
    case funds
    case holdings
    case trades
}

struct UserProfile: Decodable {
    let id: Int
    let username: String
    let email: String
}

struct TokenResponse: Decodable {
    let access: String
    let refresh: String
}

struct FundDashboardResponse: Decodable {
    let totalValue: Double
    let returnAmount: Double
    let cash: Double
    let holdingsCount: Int
    let tradePnl: Double
    let holdingsPnl: Double
    let returnPercent: Double
    let estAbvPercent: Double
    let estabDays: Int
    let todayPercent: Double

    private enum CodingKeys: String, CodingKey {
        case totalValue = "total_value"
        case returnAmount = "return_amount"
        case cash
        case holdingsCount = "holdings_count"
        case tradePnl = "trade_pnl"
        case holdingsPnl = "holdings_pnl"
        case returnPercent = "return_percent"
        case estAbvPercent = "est_abv_percent"
        case estabDays = "estab_days"
        case todayPercent = "today_percent"
    }
}

struct FundResponse: Decodable, Identifiable {
    let id: Int
    let name: String
    let spread: String?
    let risk: String
    let advisors: [String]
    let dashboard: FundDashboardResponse
}

struct GlobalDashboardResponse: Decodable {
    let totalValue: Double
    let holdingsPnl: Double
    let returnPercent: Double
    let todayPercent: Double

    private enum CodingKeys: String, CodingKey {
        case totalValue = "total_value"
        case holdingsPnl = "holdings_pnl"
        case returnPercent = "return_percent"
        case todayPercent = "today_percent"
    }
}

struct DashboardHistoryPointResponse: Decodable {
    let date: String
    let wealth: Double
    let cash: Double
    let holdings: Double
}

struct DashboardHistoryResponse: Decodable {
    let points: [DashboardHistoryPointResponse]
    let changePercent: Double

    private enum CodingKeys: String, CodingKey {
        case points
        case changePercent = "change_percent"
    }
}

struct WealthChartPoint: Identifiable {
    let id: String
    let date: Date
    let wealth: Double
}

struct StockInfo: Decodable {
    let symbol: String
    let company: String?
    let price: String?
}

struct HoldingResponse: Decodable, Identifiable {
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

struct TradeResponse: Decodable, Identifiable {
    let id: Int
    let stock: StockInfo
    let action: String
    let price: String
    let shares: Int
    let sa: Int?
    let created: String?
}

struct LoginRequest: Encodable {
    let username: String
    let password: String
}

struct APIEnvironment {
    enum HostOption: String, CaseIterable, Identifiable {
        case local = "192.168.1.6:8000"
        case klynt = "klynt.com"

        var id: String { rawValue }

        var baseURL: URL {
            switch self {
            case .local:
                return URL(string: "http://192.168.1.6:8000/api/")!
            case .klynt:
                return URL(string: "https://klynt.com/api/")!
            }
        }
    }
}

struct TokenStore {
    private enum Keys {
        static let access = "auth.accessToken"
        static let refresh = "auth.refreshToken"
    }

    func getAccessToken() -> String? { UserDefaults.standard.string(forKey: Keys.access) }

    func save(access: String, refresh: String) {
        UserDefaults.standard.set(access, forKey: Keys.access)
        UserDefaults.standard.set(refresh, forKey: Keys.refresh)
    }

    func clear() {
        UserDefaults.standard.removeObject(forKey: Keys.access)
        UserDefaults.standard.removeObject(forKey: Keys.refresh)
    }
}

enum APIError: Error, LocalizedError {
    case invalidResponse
    case missingToken
    case httpStatus(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "Invalid server response."
        case .missingToken: return "Missing access token."
        case let .httpStatus(code, body):
            return body.isEmpty ? "Request failed (\(code))." : "Request failed (\(code)): \(body)"
        }
    }
}

struct APIClient {
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

    func fetchGlobalDashboard(accessToken: String) async throws -> GlobalDashboardResponse {
        let url = endpoint("dashboard/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(GlobalDashboardResponse.self, from: data)
    }

    func fetchDashboardHistory(accessToken: String, fundId: Int?, days: Int = 90) async throws -> DashboardHistoryResponse {
        let base = endpoint("dashboard/history/")
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        var items = [URLQueryItem(name: "days", value: String(days))]
        if let fundId {
            items.append(URLQueryItem(name: "fund_id", value: String(fundId)))
        }
        components.queryItems = items
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(DashboardHistoryResponse.self, from: data)
    }

    func fetchHoldings(accessToken: String, fundId: Int?) async throws -> [HoldingResponse] {
        let base = endpoint("holdings/")
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        if let fundId { components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))] }
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
        if let fundId { components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))] }
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([TradeResponse].self, from: data)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            throw APIError.httpStatus(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
    }
}

@MainActor
final class AuthViewModel: ObservableObject {
    @Published var username = ""
    @Published var password = ""
    @Published var selectedHost: APIEnvironment.HostOption = .local
    @Published var selectedTab: AppTab = .funds
    @Published var selectedFundId: Int?
    @Published var currentUser: UserProfile?
    @Published var funds: [FundResponse] = []
    @Published var holdings: [HoldingResponse] = []
    @Published var trades: [TradeResponse] = []
    @Published var globalDashboard: GlobalDashboardResponse?
    @Published var globalHistory: [WealthChartPoint] = []
    @Published var selectedFundHistory: [WealthChartPoint] = []
    @Published var isLoading = false
    @Published var statusMessage: String?

    private let tokenStore = TokenStore()

    private var apiClient: APIClient {
        APIClient(baseURL: selectedHost.baseURL)
    }

    var isAuthenticated: Bool { tokenStore.getAccessToken() != nil }
    var hasSelectedFund: Bool { selectedFundId != nil }
    var selectedFundName: String? { funds.first(where: { $0.id == selectedFundId })?.name }
    var selectedFund: FundResponse? { funds.first(where: { $0.id == selectedFundId }) }
    var activeHistory: [WealthChartPoint] { selectedTab == .funds ? globalHistory : selectedFundHistory }

    var headerTitle: String {
        switch selectedTab {
        case .funds: return "SOULTRADER - FUNDS"
        case .holdings, .trades:
            if let selectedFundName, !selectedFundName.isEmpty { return "SOULTRADER - \(selectedFundName)" }
            return "SOULTRADER"
        }
    }

    func bootstrap() async {
        guard isAuthenticated else { statusMessage = "Not logged in."; return }
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

    func refreshAll() async {
        isLoading = true
        defer { isLoading = false }
        do {
            guard let access = tokenStore.getAccessToken() else { throw APIError.missingToken }
            async let userTask = apiClient.fetchCurrentUser(accessToken: access)
            async let fundsTask = apiClient.fetchFunds(accessToken: access)
            async let holdingsTask = apiClient.fetchHoldings(accessToken: access, fundId: selectedFundId)
            async let tradesTask = apiClient.fetchTrades(accessToken: access, fundId: selectedFundId)
            currentUser = try await userTask
            funds = try await fundsTask
            holdings = try await holdingsTask
            trades = try await tradesTask

            // Nice-to-have endpoints: never block core list data if unavailable.
            if let dashboard = try? await apiClient.fetchGlobalDashboard(accessToken: access) {
                globalDashboard = dashboard
            } else {
                globalDashboard = nil
            }

            if let history = try? await apiClient.fetchDashboardHistory(accessToken: access, fundId: nil) {
                globalHistory = mapHistoryPoints(history.points)
            } else {
                globalHistory = []
            }

            if let fundId = selectedFundId,
               let history = try? await apiClient.fetchDashboardHistory(accessToken: access, fundId: fundId) {
                selectedFundHistory = mapHistoryPoints(history.points)
            } else {
                selectedFundHistory = []
            }
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
        globalDashboard = nil
        globalHistory = []
        selectedFundHistory = []
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

    func clearSelectedFund() { selectedFundId = nil }

    private func mapHistoryPoints(_ raw: [DashboardHistoryPointResponse]) -> [WealthChartPoint] {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return raw.compactMap { point in
            guard let date = formatter.date(from: point.date) else { return nil }
            return WealthChartPoint(id: point.date, date: date, wealth: point.wealth)
        }
    }
}

struct FundSummaryCard: View {
    let fund: FundResponse

    var body: some View {
        HStack(spacing: 8) {
            pair(title: "WEALTH", value: formatCurrency(fund.dashboard.totalValue), alignment: .leading)
            pair(title: "PORTFOLIO", value: formatPercent(fund.dashboard.holdingsPnl), color: percentColor(fund.dashboard.holdingsPnl), alignment: .trailing)
            pair(title: "TODAY", value: formatPercent(fund.dashboard.todayPercent), color: percentColor(fund.dashboard.todayPercent), alignment: .trailing)
            pair(title: "P&L", value: formatPercent(fund.dashboard.returnPercent), color: percentColor(fund.dashboard.returnPercent), alignment: .trailing)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func pair(title: String, value: String, color: Color = Theme.valuePrimary, alignment: Alignment) -> some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(color)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: alignment)
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 0
        formatter.roundingMode = .halfUp
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }

    private func formatPercent(_ value: Double) -> String {
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
    }

    private func percentColor(_ value: Double) -> Color {
        let normalized = abs(value) < 0.005 ? 0.0 : value
        if normalized > 0 { return .green }
        if normalized < 0 { return .red }
        return Theme.valuePrimary
    }
}

struct GlobalSummaryCard: View {
    let dashboard: GlobalDashboardResponse

    var body: some View {
        HStack(spacing: 8) {
            pair(title: "WEALTH", value: formatCurrency(dashboard.totalValue), alignment: .leading)
            pair(title: "PORTFOLIO", value: formatPercent(dashboard.holdingsPnl), color: percentColor(dashboard.holdingsPnl), alignment: .trailing)
            pair(title: "TODAY", value: formatPercent(dashboard.todayPercent), color: percentColor(dashboard.todayPercent), alignment: .trailing)
            pair(title: "P&L", value: formatPercent(dashboard.returnPercent), color: percentColor(dashboard.returnPercent), alignment: .trailing)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func pair(title: String, value: String, color: Color = Theme.valuePrimary, alignment: Alignment) -> some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(color)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: alignment)
    }

    private func formatCurrency(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 0
        formatter.roundingMode = .halfUp
        return formatter.string(from: NSNumber(value: value)) ?? "$0"
    }

    private func formatPercent(_ value: Double) -> String {
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
    }

    private func percentColor(_ value: Double) -> Color {
        let normalized = abs(value) < 0.005 ? 0.0 : value
        if normalized > 0 { return .green }
        if normalized < 0 { return .red }
        return Theme.valuePrimary
    }
}

struct WealthChartCard: View {
    let points: [WealthChartPoint]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if points.count >= 2 {
                Chart(points) { point in
                    AreaMark(
                        x: .value("Date", point.date),
                        y: .value("Wealth", point.wealth)
                    )
                    .foregroundStyle(
                        .linearGradient(
                            colors: [Color.green.opacity(0.25), Color.clear],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )

                    LineMark(
                        x: .value("Date", point.date),
                        y: .value("Wealth", point.wealth)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(.green)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }
                .chartYScale(domain: yScale.lower...yScale.upper)
                .chartXAxis(.hidden)
                .chartYAxis {
                    AxisMarks(position: .leading, values: .stride(by: yScale.step)) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color.white.opacity(0.15))
                        AxisValueLabel {
                            if let val = value.as(Double.self) {
                                Text(shortCurrency(val))
                                    .font(.caption2)
                                    .foregroundStyle(Theme.secondaryText)
                            }
                        }
                    }
                }
                .frame(height: 120)
            } else {
                Text("No snapshot history yet.")
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryText)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .center)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func shortCurrency(_ value: Double) -> String {
        if value >= 1_000_000 {
            return String(format: "$%.1fM", value / 1_000_000)
        }
        if value >= 1_000 {
            return String(format: "$%.0fK", value / 1_000)
        }
        return String(format: "$%.0f", value)
    }

    private var yScale: (lower: Double, upper: Double, step: Double) {
        let values = points.map(\.wealth)
        guard let minValue = values.min(), let maxValue = values.max() else {
            return (0, 5_000, 5_000)
        }

        let range = maxValue - minValue
        let step = gradientStep(for: range)
        let padding = max(step * 0.2, 1_000)
        var lower = floor((minValue - padding) / step) * step
        var upper = ceil((maxValue + padding) / step) * step

        if lower == upper {
            upper += step
        }
        if lower < 0 {
            lower = 0
        }
        return (lower, upper, step)
    }

    // Match axis/grid stride to visible wealth range.
    private func gradientStep(for range: Double) -> Double {
        switch range {
        case ..<10_000:
            return 5_000
        case ..<30_000:
            return 10_000
        case ..<80_000:
            return 20_000
        case ..<200_000:
            return 50_000
        default:
            return 100_000
        }
    }
}

func configureTabBarAppearance() {
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

private func tabBarGradientImage(size: CGSize, topLeft: UIColor, bottomRight: UIColor) -> UIImage {
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

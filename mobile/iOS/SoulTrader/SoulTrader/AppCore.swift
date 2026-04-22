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
    let returnAmount: Double
    let cash: Double
    let holdingsPnl: Double
    let returnPercent: Double
    let todayPercent: Double

    private enum CodingKeys: String, CodingKey {
        case totalValue = "total_value"
        case returnAmount = "return_amount"
        case cash
        case holdingsPnl = "holdings_pnl"
        case returnPercent = "return_percent"
        case todayPercent = "today_percent"
    }

    /// Market value of stock positions (excludes cash); matches fund dashboard `total_value - cash`.
    var holdingsMarketValue: Double {
        max(0, totalValue - cash)
    }
}

extension FundDashboardResponse {
    /// Market value of stock positions (excludes cash).
    var holdingsMarketValue: Double {
        max(0, totalValue - cash)
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

struct StockPriceHistoryPointResponse: Decodable {
    let date: String
    let close: Double
}

struct StockPriceHistoryResponse: Decodable {
    let symbol: String
    let points: [StockPriceHistoryPointResponse]
}

struct StockPriceChartPoint: Identifiable {
    let id: String
    let date: Date
    let close: Double
}

struct StockInfo: Decodable {
    let symbol: String
    let company: String?
    let industry: String?
    let sector: String?
    let exchange: String?
    let price: String?
}

struct HoldingResponse: Decodable, Identifiable {
    let id: Int
    let stockId: Int
    let stock: StockInfo
    let shares: Int
    let averagePrice: String
    let discoveryName: String?
    let discoveryLogo: String?
    let discoveryComment: String?
    let discoveryExplanation: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case stockId = "stock_id"
        case stock
        case shares
        case averagePrice = "average_price"
        case discoveryName = "discovery_name"
        case discoveryLogo = "discovery_logo"
        case discoveryComment = "discovery_comment"
        case discoveryExplanation = "discovery_explanation"
    }
}

// MARK: - Holding health history (matches web holding_history health_history)

struct HoldingHealthHistoryResponse: Decodable {
    let healthHistory: [HealthHistoryRecord]

    private enum CodingKeys: String, CodingKey {
        case healthHistory = "health_history"
    }
}

struct HealthHistoryRecord: Decodable, Identifiable {
    let id: Int
    let score: Double
    let created: String?
    let meta: HealthMetaPayload?
    let confidenceScore: HealthScalar?
    let healthScore: HealthScalar?
    let valuationScore: HealthScalar?
    let piotroski: HealthScalar?
    let altmanZ: HealthScalar?
    let geminiWeight: HealthScalar?
    let geminiRec: HealthScalar?
    let geminiExplanation: HealthScalar?
    let overlayPoints: Double?
    let overlayReasons: [String]

    private enum CodingKeys: String, CodingKey {
        case id
        case score
        case created
        case meta
        case confidenceScore = "confidence_score"
        case healthScore = "health_score"
        case valuationScore = "valuation_score"
        case piotroski
        case altmanZ = "altman_z"
        case geminiWeight = "gemini_weight"
        case geminiRec = "gemini_rec"
        case geminiExplanation = "gemini_explanation"
        case overlayPoints = "overlay_points"
        case overlayReasons = "overlay_reasons"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(Int.self, forKey: .id)
        score = try c.decode(Double.self, forKey: .score)
        created = try c.decodeIfPresent(String.self, forKey: .created)
        meta = try c.decodeIfPresent(HealthMetaPayload.self, forKey: .meta)
        confidenceScore = try c.decodeIfPresent(HealthScalar.self, forKey: .confidenceScore)
        healthScore = try c.decodeIfPresent(HealthScalar.self, forKey: .healthScore)
        valuationScore = try c.decodeIfPresent(HealthScalar.self, forKey: .valuationScore)
        piotroski = try c.decodeIfPresent(HealthScalar.self, forKey: .piotroski)
        altmanZ = try c.decodeIfPresent(HealthScalar.self, forKey: .altmanZ)
        geminiWeight = try c.decodeIfPresent(HealthScalar.self, forKey: .geminiWeight)
        geminiRec = try c.decodeIfPresent(HealthScalar.self, forKey: .geminiRec)
        geminiExplanation = try c.decodeIfPresent(HealthScalar.self, forKey: .geminiExplanation)
        if let pts = try? c.decode(Double.self, forKey: .overlayPoints) {
            overlayPoints = pts
        } else if let pts = try? c.decode(String.self, forKey: .overlayPoints), let d = Double(pts) {
            overlayPoints = d
        } else {
            overlayPoints = nil
        }
        overlayReasons = try c.decodeIfPresent([String].self, forKey: .overlayReasons) ?? []
    }

    var renderKind: String {
        (meta?.render ?? "advisor").lowercased()
    }
}

struct HealthMetaPayload: Decodable {
    let render: String?
    let media: HealthMediaPayload?
}

struct HealthMediaPayload: Decodable {
    let summary: String?
}

struct HealthScalar: Decodable {
    let display: String

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if try c.decodeNil() {
            display = "—"
            return
        }
        if let s = try? c.decode(String.self) {
            display = s
            return
        }
        if let i = try? c.decode(Int.self) {
            display = String(i)
            return
        }
        if let d = try? c.decode(Double.self) {
            display = d.truncatingRemainder(dividingBy: 1) == 0 ? String(format: "%.0f", d) : String(format: "%.2f", d)
            return
        }
        if let b = try? c.decode(Bool.self) {
            display = b ? "Yes" : "No"
            return
        }
        display = "—"
    }
}

struct TradeResponse: Decodable, Identifiable {
    let id: Int
    let stock: StockInfo
    let action: String
    let price: String
    let shares: Int
    /// Average cost basis at time of SELL; null for BUY rows.
    let cost: String?
    let explanation: String?
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

    func fetchStockPriceHistory(accessToken: String, symbol: String, period: String = "2mo") async throws -> StockPriceHistoryResponse {
        var components = URLComponents(url: endpoint("stocks/price_history/"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "symbol", value: symbol),
            URLQueryItem(name: "period", value: period),
        ]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(StockPriceHistoryResponse.self, from: data)
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

    func fetchHoldingHealthHistory(accessToken: String, fundId: Int, stockId: Int) async throws -> HoldingHealthHistoryResponse {
        let path = "holdings/\(stockId)/health_history/"
        var components = URLComponents(url: endpoint(path), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(HoldingHealthHistoryResponse.self, from: data)
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
    private enum RememberedLoginKeys {
        static let username = "remembered_login_username"
        static let password = "remembered_login_password"
        static let host = "remembered_login_host"
    }

    init() {
        loadRememberedLoginInputs()
    }

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
            saveRememberedLoginInputs()
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

    func loadHoldingHealthHistory(stockId: Int) async -> [HealthHistoryRecord] {
        guard let access = tokenStore.getAccessToken(),
              let fundId = selectedFundId else { return [] }
        do {
            let response = try await apiClient.fetchHoldingHealthHistory(
                accessToken: access,
                fundId: fundId,
                stockId: stockId
            )
            return response.healthHistory
        } catch {
            return []
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
        selectedTab = .funds
        statusMessage = "Logged out."
    }

    func selectFund(_ fundId: Int) async {
        selectedFundId = fundId
        selectedTab = .holdings
        await refreshAll()
    }

    func clearSelectedFund() { selectedFundId = nil }

    /// Daily closes for trade detail chart (web `holding_history` parity).
    func fetchTradeSymbolPriceHistory(symbol: String) async -> [StockPriceChartPoint] {
        guard let access = tokenStore.getAccessToken() else { return [] }
        do {
            let response = try await apiClient.fetchStockPriceHistory(accessToken: access, symbol: symbol)
            return mapStockPriceHistoryPoints(response.points)
        } catch {
            return []
        }
    }

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

    private func mapStockPriceHistoryPoints(_ raw: [StockPriceHistoryPointResponse]) -> [StockPriceChartPoint] {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        let cal = Calendar.current
        return raw.compactMap { point in
            guard let date = formatter.date(from: point.date) else { return nil }
            let day = cal.startOfDay(for: date)
            return StockPriceChartPoint(id: point.date, date: day, close: point.close)
        }
    }

    private func saveRememberedLoginInputs() {
        let defaults = UserDefaults.standard
        defaults.set(username, forKey: RememberedLoginKeys.username)
        defaults.set(password, forKey: RememberedLoginKeys.password)
        defaults.set(selectedHost.rawValue, forKey: RememberedLoginKeys.host)
    }

    private func loadRememberedLoginInputs() {
        let defaults = UserDefaults.standard
        username = defaults.string(forKey: RememberedLoginKeys.username) ?? ""
        password = defaults.string(forKey: RememberedLoginKeys.password) ?? ""
        if let hostRaw = defaults.string(forKey: RememberedLoginKeys.host),
           let host = APIEnvironment.HostOption(rawValue: hostRaw) {
            selectedHost = host
        }
    }
}

private enum SummaryMetricAlignment {
    case leading
    case trailing
}

private struct SummaryMetricItem {
    let title: String
    let value: String
    let color: Color
    let alignment: SummaryMetricAlignment
}

private struct SummaryMetricCard: View {
    let items: [SummaryMetricItem]

    var body: some View {
        HStack(spacing: 8) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                VStack(alignment: item.alignment == .leading ? .leading : .trailing, spacing: 2) {
                    Text(item.title)
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.labelAccent)
                    Text(item.value)
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(item.color)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: item.alignment == .leading ? .leading : .trailing)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }
}

struct FundSummaryCard: View {
    let fund: FundResponse

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: "WEALTH",
                value: formatCurrency(fund.dashboard.totalValue),
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "CASH",
                value: formatCurrency(fund.dashboard.cash),
                color: Theme.valuePrimary,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "PORTFOLIO",
                value: formatCurrency(fund.dashboard.holdingsMarketValue),
                color: percentColor(fund.dashboard.holdingsPnl),
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "P&L",
                value: formatPercent(fund.dashboard.returnPercent),
                color: percentColor(fund.dashboard.returnPercent),
                alignment: .trailing
            ),
        ])
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

struct FundSecondarySummaryCard: View {
    let countTitle: String
    let countValue: String
    let equityPercent: Double?
    let middleTitle: String
    let middleValue: String
    let middleColor: Color
    let todayPercent: Double?

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: countTitle,
                value: countValue,
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "EQUITY",
                value: formatPercent(equityPercent),
                color: percentColor(equityPercent),
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: middleTitle,
                value: middleValue,
                color: middleColor,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "TODAY",
                value: formatPercent(todayPercent),
                color: percentColor(todayPercent),
                alignment: .trailing
            ),
        ])
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
    }

    private func percentColor(_ value: Double?) -> Color {
        guard let value else { return Theme.valuePrimary }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        if normalized > 0 { return .green }
        if normalized < 0 { return .red }
        return Theme.valuePrimary
    }
}

struct GlobalSummaryCard: View {
    let dashboard: GlobalDashboardResponse

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: "WEALTH",
                value: formatCurrency(dashboard.totalValue),
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "CASH",
                value: formatCurrency(dashboard.cash),
                color: Theme.valuePrimary,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "PORTFOLIO",
                value: formatCurrency(dashboard.holdingsMarketValue),
                color: percentColor(dashboard.holdingsPnl),
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "P&L",
                value: formatPercent(dashboard.returnPercent),
                color: percentColor(dashboard.returnPercent),
                alignment: .trailing
            ),
        ])
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
                .chartPlotStyle { plot in
                    plot.clipShape(Rectangle())
                }
                .frame(maxWidth: .infinity, minHeight: 120, maxHeight: 120, alignment: .center)
                .clipped()
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
        .clipShape(RoundedRectangle(cornerRadius: 10))
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

/// Share price over time (trade detail); Y axis is per-share close, not portfolio wealth.
struct SharePriceChartCard: View {
    let symbol: String
    let points: [StockPriceChartPoint]
    /// Execution time (full timestamp); X is drawn at start of that local calendar day to match daily bars.
    let tradeAt: Date?
    let tradePrice: Double?

    /// Align marker with the daily close series (exchange-local day).
    private var tradeMarkerX: Date? {
        guard let t = tradeAt else { return nil }
        return Calendar.current.startOfDay(for: t)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Share price")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                Spacer()
                Text("\(symbol) · daily")
                    .font(.caption2)
                    .foregroundStyle(Theme.secondaryText)
            }

            if points.count >= 2 {
                Chart {
                    ForEach(points) { point in
                        AreaMark(
                            x: .value("Date", point.date),
                            y: .value("Close", point.close)
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
                            y: .value("Close", point.close)
                        )
                        .interpolationMethod(.catmullRom)
                        .foregroundStyle(.green)
                        .lineStyle(StrokeStyle(lineWidth: 2))
                    }

                    if let tx = tradeMarkerX, let py = tradePrice {
                        RuleMark(x: .value("Trade day", tx))
                            .foregroundStyle(Color.white.opacity(0.28))
                            .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))

                        PointMark(
                            x: .value("Trade", tx),
                            y: .value("Trade", py)
                        )
                        .symbol(.circle)
                        .symbolSize(70)
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.35), radius: 2, y: 1)
                        .accessibilityLabel("Trade at \(formattedTradeMarkerPrice(py)) on \(formattedTradeMarkerDay(tx))")
                    }
                }
                .chartXScale(domain: xDomain)
                .chartYScale(domain: yDomain)
                .chartXAxis {
                    AxisMarks(values: .stride(by: .day, count: xAxisDayStride, calendar: .current)) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color.white.opacity(0.12))
                        AxisValueLabel(centered: true) {
                            if let date = value.as(Date.self) {
                                Text(formatXAxisDate(date))
                                    .font(.caption2)
                                    .foregroundStyle(Theme.secondaryText)
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.65)
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color.white.opacity(0.15))
                        AxisValueLabel {
                            if let val = value.as(Double.self) {
                                Text(formatPriceAxis(val))
                                    .font(.caption2)
                                    .foregroundStyle(Theme.secondaryText)
                                    .lineLimit(1)
                                    .minimumScaleFactor(0.7)
                            }
                        }
                    }
                }
                .chartPlotStyle { plot in
                    plot.clipShape(Rectangle())
                }
                .frame(maxWidth: .infinity, minHeight: 132, maxHeight: 148, alignment: .center)
                .clipped()
            } else {
                Text("No price history yet.")
                    .font(.caption)
                    .foregroundStyle(Theme.secondaryText)
                    .frame(maxWidth: .infinity, minHeight: 80, alignment: .center)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func formattedTradeMarkerPrice(_ price: Double) -> String {
        if price >= 100 { return String(format: "$%.0f", price) }
        if price >= 10 { return String(format: "$%.2f", price) }
        return String(format: "$%.2f", price)
    }

    private func formattedTradeMarkerDay(_ day: Date) -> String {
        let df = DateFormatter()
        df.locale = .current
        df.setLocalizedDateFormatFromTemplate("MMM d")
        return df.string(from: day)
    }

    /// Tight X domain on data; expands if the trade falls outside the history window.
    private var xDomain: ClosedRange<Date> {
        guard let first = points.map(\.date).min(), let last = points.map(\.date).max() else {
            let n = Date()
            return n...n
        }
        var a = first
        var b = last
        if let tx = tradeMarkerX {
            if tx < a { a = tx }
            if tx > b { b = tx }
        }
        if a >= b {
            if let end = Calendar.current.date(byAdding: .second, value: 1, to: a) {
                return a...end
            }
            return a...b
        }
        return a...b
    }

    /// Day stride so we get ~4–5 ticks across the visible range (no `AxisMarkValues.explicit` on older Charts).
    private var xAxisDayStride: Int {
        let span = max(1, xSpanDays)
        return max(1, min(120, span / 4))
    }

    private var xSpanDays: Int {
        let first = xDomain.lowerBound
        let last = xDomain.upperBound
        return Calendar.current.dateComponents([.day], from: first, to: last).day ?? 0
    }

    private func formatXAxisDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = .current
        if xSpanDays > 400 {
            formatter.setLocalizedDateFormatFromTemplate("MMM yy")
        } else {
            formatter.setLocalizedDateFormatFromTemplate("MMM d")
        }
        return formatter.string(from: date)
    }

    /// Short labels so a capped tick count (~4) stays readable on a narrow Y axis.
    private func formatPriceAxis(_ value: Double) -> String {
        let v = abs(value)
        if v >= 1_000 {
            return String(format: "$%.2fK", value / 1_000)
        }
        if v >= 100 {
            return String(format: "$%.0f", value)
        }
        if v >= 10 {
            return String(format: "$%.1f", value)
        }
        if v >= 1 {
            return String(format: "$%.2f", value)
        }
        return String(format: "$%.3f", value)
    }

    private var yDomain: ClosedRange<Double> {
        var values = points.map(\.close)
        if let p = tradePrice {
            values.append(p)
        }
        guard let minValue = values.min(), let maxValue = values.max() else {
            return 0...100
        }

        let range = maxValue - minValue
        let padding = max(range * 0.08, max(range * 0.02, 0.01))
        var lower = minValue - padding
        var upper = maxValue + padding
        if lower < 0 {
            lower = 0
        }
        if lower >= upper {
            upper = lower + max(lower * 0.02, 0.01)
        }
        return lower...upper
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

import SwiftUI
import UIKit

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

struct FundResponse: Decodable, Identifiable {
    let id: Int
    let name: String
    let spread: String?
    let risk: String
    let advisors: [String]
    let dashboard: FundDashboardResponse
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
    @Published var isLoading = false
    @Published var statusMessage: String?

    private let tokenStore = TokenStore()

    private var apiClient: APIClient {
        APIClient(baseURL: selectedHost.baseURL)
    }

    var isAuthenticated: Bool { tokenStore.getAccessToken() != nil }
    var hasSelectedFund: Bool { selectedFundId != nil }
    var selectedFundName: String? { funds.first(where: { $0.id == selectedFundId })?.name }

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

    func clearSelectedFund() { selectedFundId = nil }
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

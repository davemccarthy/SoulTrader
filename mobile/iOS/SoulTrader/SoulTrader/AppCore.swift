import SwiftUI
import UIKit
import Charts
import CryptoKit
import FirebaseMessaging

let appBackground = Theme.appBackground

enum AppTab {
    case funds
    case holdings
    case trades
    case advisory
}

enum ReturnPercentMode {
    case total
    case invested
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
    let estAprPercent: Double
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
        case estAprPercent = "est_apr_percent"
        case estabDays = "estab_days"
        case todayPercent = "today_percent"
    }
}

struct FundResponse: Decodable, Identifiable {
    let id: Int
    let name: String
    /// Full sheet text: auto intro + optional admin copy.
    let description: String
    /// Admin `Profile.description` only — used for read/ack digest, not the live intro.
    let profileDescription: String
    let spread: String?
    let risk: String
    let advisors: [String]
    let dashboard: FundDashboardResponse

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case profileDescription = "profile_description"
        case spread
        case risk
        case advisors
        case dashboard
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(Int.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        profileDescription = try c.decodeIfPresent(String.self, forKey: .profileDescription) ?? ""
        spread = try c.decodeIfPresent(String.self, forKey: .spread)
        risk = try c.decode(String.self, forKey: .risk)
        advisors = try c.decode([String].self, forKey: .advisors)
        dashboard = try c.decode(FundDashboardResponse.self, forKey: .dashboard)
    }
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

    /// Return percent against currently invested capital (market value excluding cash).
    var investedReturnPercent: Double? {
        guard holdingsMarketValue > 0 else { return nil }
        return (returnAmount / holdingsMarketValue) * 100
    }
}

extension FundDashboardResponse {
    /// Market value of stock positions (excludes cash).
    var holdingsMarketValue: Double {
        max(0, totalValue - cash)
    }

    /// Return percent against currently invested capital (market value excluding cash).
    var investedReturnPercent: Double? {
        guard holdingsMarketValue > 0 else { return nil }
        return (returnAmount / holdingsMarketValue) * 100
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
    let scoring: DiscoveryScoring?

    private enum CodingKeys: String, CodingKey {
        case healthHistory = "health_history"
        case scoring
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        healthHistory = try c.decodeIfPresent([HealthHistoryRecord].self, forKey: .healthHistory) ?? []
        scoring = try c.decodeIfPresent(DiscoveryScoring.self, forKey: .scoring)
    }
}

struct HoldingHeadlinesResponse: Decodable {
    let headlines: [String]

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        headlines = try c.decodeIfPresent([String].self, forKey: .headlines) ?? []
    }

    private enum CodingKeys: String, CodingKey {
        case headlines
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

    /// Mirrors web `hasEdgarPayload`: Ex-99, media, bonuses, or penalties present in meta.
    var hasEdgarStructuredPayload: Bool {
        guard renderKind == "edgar", let meta else { return false }
        if meta.ex99?.hasStructuredContent == true { return true }
        if meta.media?.hasStructuredContent == true { return true }
        if let b = meta.bonuses, !b.isEmpty { return true }
        if let p = meta.penalties, !p.isEmpty { return true }
        return false
    }
}

struct HealthMetaPayload: Decodable {
    let render: String?
    let ex99: HealthEx99Payload?
    let media: HealthMediaPayload?
    let bonuses: [String]?
    let penalties: [String]?
}

struct HealthEx99Payload: Decodable {
    let pastPerformance: String?
    let guidance: String?
    let expectation: String?
    let marketReaction: String?
    let justifications: HealthEx99Justifications?

    enum CodingKeys: String, CodingKey {
        case pastPerformance = "past_performance"
        case guidance
        case expectation
        case marketReaction = "market_reaction"
        case justifications
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        pastPerformance = try c.decodeFlexibleOptionalString(forKey: .pastPerformance)
        guidance = try c.decodeFlexibleOptionalString(forKey: .guidance)
        expectation = try c.decodeFlexibleOptionalString(forKey: .expectation)
        marketReaction = try c.decodeFlexibleOptionalString(forKey: .marketReaction)
        justifications = try c.decodeIfPresent(HealthEx99Justifications.self, forKey: .justifications)
    }

    /// Any Ex-99 text or categorical fields worth showing.
    var hasStructuredContent: Bool {
        if let j = justifications, j.hasContent { return true }
        let top = [pastPerformance, guidance, expectation, marketReaction].compactMap { $0 }.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        return !top.isEmpty
    }
}

struct HealthEx99Justifications: Decodable {
    let pastPerformance: String?
    let guidance: String?
    let expectation: String?
    let marketReaction: String?

    enum CodingKeys: String, CodingKey {
        case pastPerformance = "past_performance"
        case guidance
        case expectation
        case marketReaction = "market_reaction"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        pastPerformance = try c.decodeFlexibleOptionalString(forKey: .pastPerformance)
        guidance = try c.decodeFlexibleOptionalString(forKey: .guidance)
        expectation = try c.decodeFlexibleOptionalString(forKey: .expectation)
        marketReaction = try c.decodeFlexibleOptionalString(forKey: .marketReaction)
    }

    var hasContent: Bool {
        [pastPerformance, guidance, expectation, marketReaction]
            .compactMap { $0 }
            .contains { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
}

struct HealthMediaPayload: Decodable {
    let summary: String?
    let sentiment: String?
    let eps: String?
    let revenue: String?
    let broker: String?
    let headlines: [String]?
    let redFlags: [String]?

    enum CodingKeys: String, CodingKey {
        case summary
        case sentiment
        case eps
        case revenue
        case broker
        case headlines
        case redFlags = "red_flags"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        summary = try c.decodeFlexibleOptionalString(forKey: .summary)
        sentiment = try c.decodeFlexibleOptionalString(forKey: .sentiment)
        eps = try c.decodeFlexibleOptionalString(forKey: .eps)
        revenue = try c.decodeFlexibleOptionalString(forKey: .revenue)
        broker = try c.decodeFlexibleOptionalString(forKey: .broker)
        headlines = try c.decodeIfPresent([String].self, forKey: .headlines)
        redFlags = try c.decodeIfPresent([String].self, forKey: .redFlags)
    }

    var hasStructuredContent: Bool {
        if let s = summary?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { return true }
        if let s = sentiment?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { return true }
        if let s = eps?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { return true }
        if let s = revenue?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { return true }
        if let s = broker?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { return true }
        if let h = headlines, !h.isEmpty { return true }
        if let r = redFlags, !r.isEmpty { return true }
        return false
    }
}

private extension KeyedDecodingContainer {
    /// Decodes string, number, or bool from optional JSON fields (SEC/meta payloads vary).
    func decodeFlexibleOptionalString(forKey key: Key) throws -> String? {
        guard contains(key) else { return nil }
        if try decodeNil(forKey: key) { return nil }
        if let s = try? decode(String.self, forKey: key) {
            return s
        }
        if let i = try? decode(Int.self, forKey: key) {
            return String(i)
        }
        if let d = try? decode(Double.self, forKey: key) {
            if d.truncatingRemainder(dividingBy: 1) == 0 {
                return String(format: "%.0f", d)
            }
            return String(d)
        }
        if let b = try? decode(Bool.self, forKey: key) {
            return b ? "true" : "false"
        }
        return nil
    }
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

struct FundAdvisorRow: Decodable, Identifiable {
    let id: Int
    let pythonClass: String
    let name: String
    let description: String
    let discoveryCount: Int
    let imageUrl: String?

    private enum CodingKeys: String, CodingKey {
        case id
        case pythonClass = "python_class"
        case name
        case description
        case discoveryCount = "discovery_count"
        case imageUrl = "image_url"
    }
}

struct FundAdvisorsResponse: Decodable {
    let fundId: Int
    let days: Int?
    let advisors: [FundAdvisorRow]

    private enum CodingKeys: String, CodingKey {
        case fundId = "fund_id"
        case days
        case advisors
    }
}

struct AdvisorScoreboardRow: Decodable, Equatable {
    let advisorId: Int
    let trades: Int
    let winners: Int
    let losers: Int
    let winRate: Double
    let gainLossPct: Double

    private enum CodingKeys: String, CodingKey {
        case advisorId = "advisor_id"
        case trades, winners, losers
        case winRate = "win_rate"
        case gainLossPct = "gain_loss_pct"
    }
}

struct FundAdvisorScoreboardResponse: Decodable {
    let fundId: Int
    let days: Int
    let advisors: [AdvisorScoreboardRow]

    private enum CodingKeys: String, CodingKey {
        case fundId = "fund_id"
        case days, advisors
    }
}

struct AdvisorDiscoveryStock: Decodable {
    let symbol: String
    let company: String
    let price: String
}

struct AdvisorDiscoveryRow: Decodable, Identifiable {
    let id: Int
    let stock: AdvisorDiscoveryStock
    let explanationLine: String
    let healthScore: Double?
    let scoring: DiscoveryScoring?

    private enum CodingKeys: String, CodingKey {
        case id
        case stock
        case explanationLine = "explanation_line"
        case healthScore = "health_score"
        case scoring
    }

    /// GRADE column: SO grade when v2, else legacy health_score.
    var listScoreText: String {
        if let scoring {
            let text = scoring.displayScoreText
            if text != "—" { return text }
        }
        return DiscoveryScoring.formatOptionalScore(healthScore)
    }
}

struct AdvisorDiscoveriesResponse: Decodable {
    let advisorId: Int
    let days: Int?
    let discoveries: [AdvisorDiscoveryRow]

    private enum CodingKeys: String, CodingKey {
        case advisorId = "advisor_id"
        case days
        case discoveries
    }
}

struct DiscoveryDetailAdvisor: Decodable {
    let id: Int
    let name: String
    let logoUrl: String?

    private enum CodingKeys: String, CodingKey {
        case id, name
        case logoUrl = "logo_url"
    }
}

// MARK: - Discovery scoring (API `discovery_scoring_context`, v2 assessment + v1 health fallback)

struct DiscoveryScoringRating: Decodable {
    let letter: String
    let label: String?

    var displayLetter: String {
        let l = letter.trimmingCharacters(in: .whitespacesAndNewlines)
        return l.isEmpty ? "—" : l
    }
}

struct DiscoveryScoringSummary: Decodable {
    let assessmentScore: Double?
    let discoveryWeightDisplay: String?
    let grade: DiscoveryScoringRating?
    let adjustedGrade: DiscoveryScoringRating?

    private enum CodingKeys: String, CodingKey {
        case assessmentScore = "assessment_score"
        case discoveryWeightDisplay = "discovery_weight_display"
        case grade
        case adjustedGrade = "adjusted_grade"
    }
}

struct DiscoveryScoringComponent: Decodable, Identifiable {
    let key: String
    let label: String
    let weightPercent: Int
    let score: Double?

    var id: String { key }

    private enum CodingKeys: String, CodingKey {
        case key, label, score
        case weightPercent = "weight_percent"
    }
}

struct DiscoveryScoringRiskFloors: Decodable {
    let minStability: String?
    let minOpportunity: String?
    let soFloorDisplay: String?
    let soCompositeFloor: String?

    private enum CodingKeys: String, CodingKey {
        case minStability = "min_stability"
        case minOpportunity = "min_opportunity"
        case soFloorDisplay = "so_floor_display"
        case soCompositeFloor = "so_composite_floor"
    }
}

struct DiscoveryScoring: Decodable {
    let source: String?
    let compositeScore: Double?
    let adjustedScore: Double?
    let discoveryWeight: Double?
    let headlineDisplay: String?
    let summary: DiscoveryScoringSummary?
    let components: [DiscoveryScoringComponent]
    let soGrade: String?
    let soGradePair: String?
    let stability: Double?
    let opportunity: Double?
    let opportunityAdjusted: Double?
    let stabilityGrade: DiscoveryScoringRating?
    let opportunityGrade: DiscoveryScoringRating?
    let opportunityAdjustedGrade: DiscoveryScoringRating?
    let showOpportunityUpgrade: Bool?
    let opportunityUpgradeDisplay: String?
    let interpretation: String?
    let riskMatrix: [String: String]?
    let riskFloors: [String: DiscoveryScoringRiskFloors]?

    private enum CodingKeys: String, CodingKey {
        case source
        case compositeScore = "composite_score"
        case adjustedScore = "adjusted_score"
        case discoveryWeight = "discovery_weight"
        case headlineDisplay = "headline_display"
        case summary, components
        case soGrade = "so_grade"
        case soGradePair = "so_grade_pair"
        case stability, opportunity
        case opportunityAdjusted = "opportunity_adjusted"
        case stabilityGrade = "stability_grade"
        case opportunityGrade = "opportunity_grade"
        case opportunityAdjustedGrade = "opportunity_adjusted_grade"
        case showOpportunityUpgrade = "show_opportunity_upgrade"
        case opportunityUpgradeDisplay = "opportunity_upgrade_display"
        case interpretation
        case riskMatrix = "risk_matrix"
        case riskFloors = "risk_floors"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        source = try c.decodeIfPresent(String.self, forKey: .source)
        compositeScore = try c.decodeIfPresent(Double.self, forKey: .compositeScore)
        adjustedScore = try c.decodeIfPresent(Double.self, forKey: .adjustedScore)
        discoveryWeight = try c.decodeIfPresent(Double.self, forKey: .discoveryWeight)
        headlineDisplay = try c.decodeIfPresent(String.self, forKey: .headlineDisplay)
        summary = try c.decodeIfPresent(DiscoveryScoringSummary.self, forKey: .summary)
        components = try c.decodeIfPresent([DiscoveryScoringComponent].self, forKey: .components) ?? []
        soGrade = try c.decodeIfPresent(String.self, forKey: .soGrade)
        soGradePair = try c.decodeIfPresent(String.self, forKey: .soGradePair)
        stability = try c.decodeIfPresent(Double.self, forKey: .stability)
        opportunity = try c.decodeIfPresent(Double.self, forKey: .opportunity)
        opportunityAdjusted = try c.decodeIfPresent(Double.self, forKey: .opportunityAdjusted)
        stabilityGrade = try c.decodeIfPresent(DiscoveryScoringRating.self, forKey: .stabilityGrade)
        opportunityGrade = try c.decodeIfPresent(DiscoveryScoringRating.self, forKey: .opportunityGrade)
        opportunityAdjustedGrade = try c.decodeIfPresent(
            DiscoveryScoringRating.self,
            forKey: .opportunityAdjustedGrade
        )
        showOpportunityUpgrade = try c.decodeIfPresent(Bool.self, forKey: .showOpportunityUpgrade)
        opportunityUpgradeDisplay = try c.decodeIfPresent(String.self, forKey: .opportunityUpgradeDisplay)
        interpretation = try c.decodeIfPresent(String.self, forKey: .interpretation)
        riskMatrix = try c.decodeIfPresent([String: String].self, forKey: .riskMatrix)
        riskFloors = try c.decodeIfPresent(
            [String: DiscoveryScoringRiskFloors].self,
            forKey: .riskFloors
        )
    }

    var isV2: Bool { source == "v2" }
    var isV1: Bool { source == "v1" }

    /// Composite SO hero grade (e.g. C+) — matches web assessment header.
    var displayGradeText: String? {
        if let grade = soGrade?.trimmingCharacters(in: .whitespacesAndNewlines), !grade.isEmpty {
            return grade
        }
        return nil
    }

    private static let gradeRank: [String: Int] = [
        "A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1,
    ]

    /// Show Upgrade row only when weight lifted opportunity by at least one letter.
    var showsOpportunityUpgrade: Bool {
        if let showOpportunityUpgrade {
            return showOpportunityUpgrade
        }
        guard let base = opportunityGrade?.letter.uppercased(),
              let adj = opportunityAdjustedGrade?.letter.uppercased(),
              let baseRank = Self.gradeRank[base],
              let adjRank = Self.gradeRank[adj] else {
            return false
        }
        return adjRank > baseRank
    }

    /// Upgrade score cell — adjusted opportunity only (no weight prefix on narrow layouts).
    var opportunityUpgradeDisplayText: String {
        DiscoveryScoring.formatOptionalScore(opportunityAdjusted)
    }

    /// List/header: SO grade when v2, else legacy numeric headline.
    var displayScoreText: String {
        if let grade = displayGradeText {
            return grade
        }
        if let headline = headlineDisplay?.trimmingCharacters(in: .whitespacesAndNewlines),
           !headline.isEmpty, headline != "—" {
            return headline
        }
        if let adjusted = adjustedScore {
            return DiscoveryScoring.formatScore(adjusted)
        }
        if let composite = compositeScore {
            return DiscoveryScoring.formatScore(composite)
        }
        return "—"
    }

    static func formatScore(_ value: Double) -> String {
        if abs(value) < 1e-9 { return "AVOID" }
        return String(format: "%.1f", value)
    }

    static func formatOptionalScore(_ value: Double?) -> String {
        guard let value else { return "—" }
        return formatScore(value)
    }
}

struct DiscoveryDetailResponse: Decodable {
    let id: Int
    let explanation: String
    let discoveryPrice: String?
    let created: String?
    let stock: StockInfo
    let advisor: DiscoveryDetailAdvisor
    let health: HealthHistoryRecord?
    let scoring: DiscoveryScoring?

    private enum CodingKeys: String, CodingKey {
        case id, explanation, created, stock, advisor, health, scoring
        case discoveryPrice = "discovery_price"
    }
}

struct DiscoveryDetailNav: Hashable {
    let discoveryId: Int
}

struct AdvisorNav: Hashable {
    let id: Int
    let name: String
    let description: String
    let imageUrl: String?
}

struct LoginRequest: Encodable {
    let username: String
    let password: String
}

struct APIEnvironment {
    /// When `false`, login hides the host row and all API calls use `klynt.com` (LAN host is ignored).
    static let showHostCredential = true

    enum HostOption: String, CaseIterable, Identifiable {
        case local1 = "192.168.1.21:8000"
        case local2 = "192.168.1.6:8000"
        case klynt = "klynt.com"

        var id: String { rawValue }

        var baseURL: URL {
            switch self {
            case .local1:
                return URL(string: "http://192.168.1.21:8000/api/")!
            case .local2:
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

private struct PushDeviceRegisterBody: Encodable {
    let token: String
    let platform: String
    let environment: String
}

private struct PushDeviceUnregisterBody: Encodable {
    let token: String
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
        // Avoid stale fund payloads (e.g. missing `description`) from URLSession disk cache.
        request.cachePolicy = .reloadIgnoringLocalCacheData
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

    func fetchHoldingHeadlines(accessToken: String, fundId: Int, stockId: Int) async throws -> HoldingHeadlinesResponse {
        let path = "holdings/\(stockId)/headlines/"
        var components = URLComponents(url: endpoint(path), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "fund_id", value: String(fundId))]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(HoldingHeadlinesResponse.self, from: data)
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

    func fetchFundAdvisors(accessToken: String, fundId: Int, days: Int = 30) async throws -> FundAdvisorsResponse {
        var components = URLComponents(url: endpoint("funds/advisors/"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "fund_id", value: String(fundId)),
            URLQueryItem(name: "days", value: String(days)),
        ]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(FundAdvisorsResponse.self, from: data)
    }

    func fetchFundAdvisorScoreboard(accessToken: String, fundId: Int, days: Int = 30) async throws -> FundAdvisorScoreboardResponse {
        var components = URLComponents(url: endpoint("funds/advisors/scoreboard/"), resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "fund_id", value: String(fundId)),
            URLQueryItem(name: "days", value: String(days)),
        ]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(FundAdvisorScoreboardResponse.self, from: data)
    }

    func fetchAdvisorDiscoveries(accessToken: String, advisorId: Int, days: Int = 30) async throws -> AdvisorDiscoveriesResponse {
        var components = URLComponents(url: endpoint("advisors/\(advisorId)/discoveries/"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "days", value: String(days))]
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(AdvisorDiscoveriesResponse.self, from: data)
    }

    func fetchDiscoveryDetail(accessToken: String, discoveryId: Int) async throws -> DiscoveryDetailResponse {
        let url = endpoint("discoveries/\(discoveryId)/")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(DiscoveryDetailResponse.self, from: data)
    }

    func registerPushDevice(accessToken: String, fcmToken: String, platform: String, environment: String) async throws {
        let url = endpoint("push/devices/")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(
            PushDeviceRegisterBody(token: fcmToken, platform: platform, environment: environment)
        )
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
    }

    func unregisterPushDevice(accessToken: String, fcmToken: String) async throws {
        let url = endpoint("push/devices/")
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        request.httpBody = try JSONEncoder().encode(PushDeviceUnregisterBody(token: fcmToken))
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        if http.statusCode == 204 {
            return
        }
        guard (200...299).contains(http.statusCode) else {
            throw APIError.httpStatus(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
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
    @Published var selectedHost: APIEnvironment.HostOption = .local1
    @Published var selectedTab: AppTab = .funds
    @Published var selectedFundId: Int?
    @Published var currentUser: UserProfile?
    @Published var funds: [FundResponse] = []
    @Published var holdings: [HoldingResponse] = []
    @Published var trades: [TradeResponse] = []
    @Published var globalDashboard: GlobalDashboardResponse?
    @Published var selectedFundHistory: [WealthChartPoint] = []
    @Published var isLoading = false
    @Published var statusMessage: String?
    @Published var returnPercentMode: ReturnPercentMode = .total

    /// Holdings fund-description sheet: "Dismiss all" skips auto-present until logout or cold launch.
    private var fundDescriptionDismissAllSession = false

    private let tokenStore = TokenStore()
    private enum RememberedLoginKeys {
        static let username = "remembered_login_username"
        static let password = "remembered_login_password"
        static let host = "remembered_login_host"
    }

    private enum PushKeys {
        static let lastRegisteredFCM = "push.last_registered_fcm_token"
    }

    /// Per-user, per-fund SHA256 (hex) of last acknowledged admin `profile_description` (normalized).
    private enum FundDescriptionAckKeys {
        static let digestByUserFund = "fund_description_ack_sha256_v2"
    }

    init() {
        loadRememberedLoginInputs()
    }

    private var apiClient: APIClient {
        APIClient(baseURL: effectiveAPIHost.baseURL)
    }

    /// Resolved host for requests: locked to production when the login host picker is hidden.
    private var effectiveAPIHost: APIEnvironment.HostOption {
        APIEnvironment.showHostCredential ? selectedHost : .klynt
    }

    /// Resolved API root (honours `APIEnvironment.showHostCredential` lock to klynt).
    var apiBaseURL: URL { effectiveAPIHost.baseURL }

    var isAuthenticated: Bool { tokenStore.getAccessToken() != nil }
    var hasSelectedFund: Bool { selectedFundId != nil }
    var selectedFundName: String? { funds.first(where: { $0.id == selectedFundId })?.name }
    var selectedFund: FundResponse? { funds.first(where: { $0.id == selectedFundId }) }
    var totalPercentTitle: String {
        switch returnPercentMode {
        case .total: return "PROFIT"
        case .invested: return "INVST"
        }
    }

    var headerTitle: String {
        switch selectedTab {
        case .funds: return "SOULTRADER - FUNDS"
        case .holdings, .trades, .advisory:
            if let selectedFundName, !selectedFundName.isEmpty { return "SOULTRADER - \(selectedFundName)" }
            return "SOULTRADER"
        }
    }

    func bootstrap() async {
        guard isAuthenticated else { statusMessage = "Not logged in."; return }
        await refreshAll()
    }

    func toggleReturnPercentMode() {
        returnPercentMode = (returnPercentMode == .total) ? .invested : .total
    }

    /// Whether the fund description sheet may auto-open (Holdings): not session-dismiss-all,
    /// and admin `profile_description` digest differs from last "Got it" for this user + fund.
    func shouldAutoPresentFundDescription(for fundId: Int, profileDescription: String) -> Bool {
        guard !fundDescriptionDismissAllSession else { return false }
        guard let userId = currentUser?.id else { return false }
        let digest = Self.fundProfileDescriptionAckDigest(profileDescription)
        let key = Self.fundDescriptionAckStorageKey(userId: userId, fundId: fundId)
        let stored = loadFundDescriptionAckDigestMap()[key]
        return stored != digest
    }

    /// Persist digest of acknowledged admin copy so auto-present skips until that text changes.
    func markFundDescriptionAcknowledged(fundId: Int, profileDescription: String) {
        guard let userId = currentUser?.id else { return }
        let digest = Self.fundProfileDescriptionAckDigest(profileDescription)
        let key = Self.fundDescriptionAckStorageKey(userId: userId, fundId: fundId)
        var map = loadFundDescriptionAckDigestMap()
        map[key] = digest
        UserDefaults.standard.set(map, forKey: FundDescriptionAckKeys.digestByUserFund)
    }

    func dismissAllFundDescriptionsForSession() {
        fundDescriptionDismissAllSession = true
    }

    private func loadFundDescriptionAckDigestMap() -> [String: String] {
        UserDefaults.standard.dictionary(forKey: FundDescriptionAckKeys.digestByUserFund) as? [String: String] ?? [:]
    }

    private static func fundDescriptionAckStorageKey(userId: Int, fundId: Int) -> String {
        "\(userId)|\(fundId)"
    }

    private static func normalizedFundDescriptionForAcknowledgment(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func fundProfileDescriptionAckDigest(_ profileDescription: String) -> String {
        sha256HexDigest(of: normalizedFundDescriptionForAcknowledgment(profileDescription))
    }

    private static func sha256HexDigest(of string: String) -> String {
        let digest = SHA256.hash(data: Data(string.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    func totalPercentValue(for dashboard: GlobalDashboardResponse) -> Double? {
        switch returnPercentMode {
        case .total:
            return dashboard.returnPercent
        case .invested:
            return dashboard.investedReturnPercent
        }
    }

    func totalPercentValue(for dashboard: FundDashboardResponse) -> Double? {
        switch returnPercentMode {
        case .total:
            return dashboard.returnPercent
        case .invested:
            return dashboard.investedReturnPercent
        }
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

            if let fundId = selectedFundId,
               let history = try? await apiClient.fetchDashboardHistory(accessToken: access, fundId: fundId) {
                selectedFundHistory = mapHistoryPoints(history.points)
            } else {
                selectedFundHistory = []
            }
            statusMessage = "Data refreshed."
            await syncFCMTokenWithServerIfAuthenticated()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func loadHoldingHealthHistory(stockId: Int) async -> (history: [HealthHistoryRecord], scoring: DiscoveryScoring?) {
        guard let access = tokenStore.getAccessToken(),
              let fundId = selectedFundId else { return ([], nil) }
        do {
            let response = try await apiClient.fetchHoldingHealthHistory(
                accessToken: access,
                fundId: fundId,
                stockId: stockId
            )
            return (response.healthHistory, response.scoring)
        } catch {
            return ([], nil)
        }
    }

    func loadHoldingHeadlines(stockId: Int) async -> [String] {
        guard let access = tokenStore.getAccessToken(),
              let fundId = selectedFundId else { return [] }
        do {
            let response = try await apiClient.fetchHoldingHeadlines(
                accessToken: access,
                fundId: fundId,
                stockId: stockId
            )
            return response.headlines
        } catch {
            return []
        }
    }

    func fetchFundAdvisors(days: Int = 30) async throws -> FundAdvisorsResponse {
        guard let access = tokenStore.getAccessToken(), let fundId = selectedFundId else { throw APIError.missingToken }
        return try await apiClient.fetchFundAdvisors(accessToken: access, fundId: fundId, days: days)
    }

    func fetchFundAdvisorScoreboard(days: Int = 30) async throws -> FundAdvisorScoreboardResponse {
        guard let access = tokenStore.getAccessToken(), let fundId = selectedFundId else { throw APIError.missingToken }
        return try await apiClient.fetchFundAdvisorScoreboard(accessToken: access, fundId: fundId, days: days)
    }

    func fetchAdvisorDiscoveries(advisorId: Int, days: Int = 30) async throws -> AdvisorDiscoveriesResponse {
        guard let access = tokenStore.getAccessToken() else { throw APIError.missingToken }
        return try await apiClient.fetchAdvisorDiscoveries(accessToken: access, advisorId: advisorId, days: days)
    }

    func fetchDiscoveryDetail(discoveryId: Int) async throws -> DiscoveryDetailResponse {
        guard let access = tokenStore.getAccessToken() else { throw APIError.missingToken }
        return try await apiClient.fetchDiscoveryDetail(accessToken: access, discoveryId: discoveryId)
    }

    /// Registers the FCM token with Django (`POST /api/push/devices/`). Safe to call repeatedly.
    func registerPushDeviceWithServer(fcmToken: String) async {
        guard let access = tokenStore.getAccessToken() else { return }
        let env = Self.pushEnvironmentLabel()
        do {
            try await apiClient.registerPushDevice(
                accessToken: access,
                fcmToken: fcmToken,
                platform: "ios",
                environment: env
            )
            UserDefaults.standard.set(fcmToken, forKey: PushKeys.lastRegisteredFCM)
        } catch {
            // Non-fatal: push plumbing should not block the app.
        }
    }

    private func syncFCMTokenWithServerIfAuthenticated() async {
        guard isAuthenticated else { return }
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            Messaging.messaging().token { token, _ in
                Task { @MainActor in
                    if let token, !token.isEmpty {
                        await self.registerPushDeviceWithServer(fcmToken: token)
                    }
                    continuation.resume()
                }
            }
        }
    }

    private static func pushEnvironmentLabel() -> String {
        #if DEBUG
        "sandbox"
        #else
        "production"
        #endif
    }

    func logout() {
        let access = tokenStore.getAccessToken()
        let lastFCM = UserDefaults.standard.string(forKey: PushKeys.lastRegisteredFCM)
        let host = effectiveAPIHost
        if let access, let lastFCM {
            Task {
                try? await APIClient(baseURL: host.baseURL).unregisterPushDevice(accessToken: access, fcmToken: lastFCM)
            }
        }
        UserDefaults.standard.removeObject(forKey: PushKeys.lastRegisteredFCM)
        tokenStore.clear()
        currentUser = nil
        selectedFundId = nil
        fundDescriptionDismissAllSession = false
        funds = []
        holdings = []
        trades = []
        globalDashboard = nil
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
        let hostToSave = APIEnvironment.showHostCredential ? selectedHost : .klynt
        defaults.set(hostToSave.rawValue, forKey: RememberedLoginKeys.host)
    }

    private func loadRememberedLoginInputs() {
        let defaults = UserDefaults.standard
        username = defaults.string(forKey: RememberedLoginKeys.username) ?? ""
        password = defaults.string(forKey: RememberedLoginKeys.password) ?? ""
        if APIEnvironment.showHostCredential,
           let hostRaw = defaults.string(forKey: RememberedLoginKeys.host),
           let host = APIEnvironment.HostOption(rawValue: hostRaw) {
            selectedHost = host
        } else {
            selectedHost = .klynt
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
    var onTap: (() -> Void)? = nil
}

private struct SummaryMetricCard: View {
    let items: [SummaryMetricItem]

    var body: some View {
        HStack(spacing: 8) {
            ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                VStack(alignment: item.alignment == .leading ? .leading : .trailing, spacing: Theme.metricSpacing) {
                    Text(item.title)
                        .appStyle(.metricLabel)
                    Text(item.value)
                        .appStyle(.metricValue, color: item.color)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: item.alignment == .leading ? .leading : .trailing)
                .contentShape(Rectangle())
                .onTapGesture {
                    item.onTap?()
                }
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius))
    }
}

struct FundSummaryCard: View {
    let fund: FundResponse
    let totalPercentTitle: String
    let totalPercentValue: Double?
    let onTap: (() -> Void)?

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: "PORTFOLIO",
                value: Theme.formatCompactCurrency(fund.dashboard.totalValue),
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "CASH",
                value: Theme.formatCompactCurrency(fund.dashboard.cash),
                color: Theme.valuePrimary,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "HOLDINGS",
                value: Theme.formatCompactCurrency(fund.dashboard.holdingsMarketValue),
                color: Theme.signedColor(for: fund.dashboard.holdingsPnl),
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: totalPercentTitle,
                value: formatPercent(totalPercentValue),
                color: Theme.signedColor(for: totalPercentValue),
                alignment: .trailing
            ),
        ])
        .contentShape(Rectangle())
        .onTapGesture {
            onTap?()
        }
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
    }

}

/// Advisory scoreboard (lookback winners/losers); shown below advisor list on Advisory tab.
struct AdvisoryScoreboardCard: View {
    let advisorCount: Int
    let winners: Int
    let losers: Int
    let lookbackDays: Int
    var onTapLookback: (() -> Void)? = nil

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: "ADVISORS",
                value: String(advisorCount),
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "WINNERS",
                value: String(winners),
                color: Theme.positive,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "LOSERS",
                value: String(losers),
                color: Theme.negative,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "LOOKBACK",
                value: "\(lookbackDays)d",
                color: Theme.secondaryText,
                alignment: .trailing,
                onTap: onTapLookback
            ),
        ])
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
                color: Theme.signedColor(for: equityPercent),
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
                color: Theme.signedColor(for: todayPercent),
                alignment: .trailing
            ),
        ])
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
    }

}

struct GlobalSummaryCard: View {
    let dashboard: GlobalDashboardResponse
    let totalPercentTitle: String
    let totalPercentValue: Double?
    let onTap: (() -> Void)?

    var body: some View {
        SummaryMetricCard(items: [
            SummaryMetricItem(
                title: "PORTFOLIO",
                value: Theme.formatCompactCurrency(dashboard.totalValue),
                color: Theme.valuePrimary,
                alignment: .leading
            ),
            SummaryMetricItem(
                title: "CASH",
                value: Theme.formatCompactCurrency(dashboard.cash),
                color: Theme.valuePrimary,
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: "HOLDINGS",
                value: Theme.formatCompactCurrency(dashboard.holdingsMarketValue),
                color: Theme.signedColor(for: dashboard.holdingsPnl),
                alignment: .trailing
            ),
            SummaryMetricItem(
                title: totalPercentTitle,
                value: formatPercent(totalPercentValue),
                color: Theme.signedColor(for: totalPercentValue),
                alignment: .trailing
            ),
        ])
        .contentShape(Rectangle())
        .onTapGesture {
            onTap?()
        }
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        let normalized = abs(value) < 0.005 ? 0.0 : value
        return String(format: "%.2f%%", normalized)
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
                                Text(Theme.formatCompactCurrency(val))
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

/// Market graph over time; Y axis is per-share close, not portfolio wealth.
struct MarketGraphCard: View {
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

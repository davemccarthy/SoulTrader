import SwiftUI

struct HoldingsView: View {
    @ObservedObject var viewModel: AuthViewModel
    @State private var path = NavigationPath()
    @State private var fundDescriptionSheetPresented = false

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 8) {
                if let fund = viewModel.selectedFund {
                    FundSummaryCard(
                        fund: fund,
                        totalPercentTitle: viewModel.totalPercentTitle,
                        totalPercentValue: viewModel.totalPercentValue(for: fund.dashboard),
                        onTap: {
                            viewModel.toggleReturnPercentMode()
                        }
                    )
                        .padding(.horizontal, 6)
                        .padding(.top, 6)
                }

                List {
                    WealthChartCard(points: viewModel.selectedFundHistory)
                        .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)

                    if let fund = viewModel.selectedFund {
                        FundSecondarySummaryCard(
                            countTitle: "STOCKS",
                            countValue: String(viewModel.holdings.count),
                            equityPercent: equityPercent(
                                totalValue: fund.dashboard.totalValue,
                                portfolioValue: fund.dashboard.holdingsMarketValue
                            ),
                            middleTitle: "HOLD",
                            middleValue: formatPercent(fund.dashboard.holdingsPnl),
                            middleColor: Theme.signedColor(for: fund.dashboard.holdingsPnl),
                            todayPercent: fund.dashboard.todayPercent
                        )
                            .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                    }

                    if viewModel.holdings.isEmpty {
                        VStack(spacing: 8) {
                            Text("No holdings to show.")
                                .appStyle(.emptyStateTitle)
                            Text("Select a fund with holdings on the FUNDS tab.")
                                .appStyle(.emptyStateMessage)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 24)
                        .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                    } else {
                        ForEach(viewModel.holdings) { holding in
                            Button {
                                path.append(holding.id)
                            } label: {
                                HStack(spacing: 12) {
                                    imageTickerPair(symbol: holding.stock.symbol)
                                    middleCompanyDiscoveryPair(holding: holding)
                                    Spacer()
                                    pricePnlPair(holding: holding)
                                }
                                .padding(.vertical, 4)
                                .padding(.horizontal, 6)
                                .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
                            }
                            .buttonStyle(.plain)
                            .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                            .listRowBackground(Color.clear)
                        }
                    }
                }
                .scrollContentBackground(.hidden)
                .scrollIndicators(.hidden)
                .contentMargins(.horizontal, 0, for: .scrollContent)
                .contentMargins(.top, 0, for: .scrollContent)
                .background(Theme.appBackground)
            }
            .background(Theme.appBackground)
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: Int.self) { holdingId in
                if let holding = viewModel.holdings.first(where: { $0.id == holdingId }) {
                    HoldingDetailView(holding: holding, baseURL: viewModel.apiBaseURL, viewModel: viewModel)
                } else {
                    Text("This holding is no longer available.")
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                }
            }
            .sheet(isPresented: $fundDescriptionSheetPresented) {
                Group {
                    if let fund = viewModel.selectedFund {
                        FundDescriptionSheetContent(
                            fundName: fund.name,
                            descriptionText: fund.description,
                                onGotIt: {
                                viewModel.markFundDescriptionAcknowledged(
                                    fundId: fund.id,
                                    profileDescription: fund.profileDescription
                                )
                                fundDescriptionSheetPresented = false
                            },
                            onDismissAll: {
                                viewModel.dismissAllFundDescriptionsForSession()
                                fundDescriptionSheetPresented = false
                            }
                        )
                        .presentationDetents([.medium, .large])
                        .presentationDragIndicator(.visible)
                    }
                }
            }
            .onAppear {
                maybeAutoPresentFundDescription()
            }
            // `onAppear` often runs before `selectFund`'s `refreshAll()` finishes; description arrives later.
            .onChange(of: selectedFundDescriptionToken) { _, _ in
                maybeAutoPresentFundDescription()
            }
            .task(id: selectedFundDescriptionToken) {
                // Brief defer so funds array is stable after async refresh.
                try? await Task.sleep(for: .milliseconds(80))
                maybeAutoPresentFundDescription()
            }
        }
    }

    /// Updates when selected fund or admin copy changes (not the live intro stats).
    private var selectedFundDescriptionToken: String {
        guard let id = viewModel.selectedFundId,
              let f = viewModel.funds.first(where: { $0.id == id }) else {
            return ""
        }
        return "\(id)|\(f.profileDescription)"
    }

    private func trimmedFundDescription(_ fund: FundResponse) -> String {
        fund.description.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Auto-open sheet when composed text is non-empty and admin-copy digest differs from last "Got it".
    private func maybeAutoPresentFundDescription() {
        guard let fund = viewModel.selectedFund else { return }
        let text = trimmedFundDescription(fund)
        guard !text.isEmpty else { return }
        guard viewModel.shouldAutoPresentFundDescription(
            for: fund.id,
            profileDescription: fund.profileDescription
        ) else { return }
        fundDescriptionSheetPresented = true
    }

    private func imageTickerPair(symbol: String) -> some View {
        VStack(spacing: 4) {
            AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
                image.resizable().scaledToFit()
            } placeholder: {
                RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
            }
            .frame(width: 26, height: 26)

            Text(symbol)
                .appStyle(.tickerSymbol)
                .lineLimit(1)
        }
        .frame(width: 50, alignment: .leading)
    }

    /// Matches advisory discovery rows: company (up to 2 lines) + first explanation snippet (2 lines), else cost basis.
    private func middleCompanyDiscoveryPair(holding: HoldingResponse) -> some View {
        let avgDecimal = decimal(from: holding.averagePrice) ?? 0
        let sharesDecimal = Decimal(holding.shares)
        let investment = sharesDecimal * avgDecimal
        let expl = holdingListExplanationLine(holding)

        return VStack(alignment: .leading, spacing: 3) {
            Text(holding.stock.company ?? holding.stock.symbol)
                .appStyle(.listHeadline)
                .lineLimit(2)
                .truncationMode(.tail)

            if let expl, !expl.isEmpty {
                Text(expl)
                    .appStyle(.listSubline)
                    .lineLimit(2)
            } else {
                Text(formatCurrency(investment))
                    .appStyle(.inlineMetricValue)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Prefer API `discovery_comment`; else first non-URL text segment from `discovery_explanation` (pipe-separated).
    private func holdingListExplanationLine(_ holding: HoldingResponse) -> String? {
        if let c = holding.discoveryComment?.trimmingCharacters(in: .whitespacesAndNewlines), !c.isEmpty {
            return c
        }
        guard let raw = holding.discoveryExplanation?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        let normalized = raw.split(whereSeparator: \.isNewline).joined(separator: " ")
        let segments = normalized.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        for segment in segments {
            if segment.hasPrefix("http://") || segment.hasPrefix("https://") {
                continue
            }
            let lower = segment.lowercased()
            if lower.hasPrefix("article:") {
                let parts = segment.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count > 1 {
                    let title = parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
                    if !title.isEmpty {
                        return String(title)
                    }
                }
                continue
            }
            return segment
        }
        return nil
    }

    private func pricePnlPair(holding: HoldingResponse) -> some View {
        let priceDecimal = decimal(from: holding.stock.price)
        let avgDecimal = decimal(from: holding.averagePrice)
        let pnlPercent = computePnlPercent(price: priceDecimal, average: avgDecimal)
        return VStack(alignment: .trailing, spacing: Theme.metricSpacing) {
            Text(formatCurrency(priceDecimal))
                .appStyle(.metricValue)
                .lineLimit(1)

            Text(formatPercent(pnlPercent))
                .appStyle(.inlineMetricValue, color: Theme.signedColor(for: pnlPercent))
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

    private func equityPercent(totalValue: Double, portfolioValue: Double) -> Double? {
        guard totalValue > 0 else { return nil }
        return (portfolioValue / totalValue) * 100
    }
}

// MARK: - Fund description sheet (Holdings only)

private struct FundDescriptionSheetContent: View {
    let fundName: String
    let descriptionText: String
    let onGotIt: () -> Void
    let onDismissAll: () -> Void

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    Text(descriptionText)
                        .font(.body)
                        .foregroundStyle(Theme.valuePrimary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.top, 8)
                }
                .frame(maxWidth: .infinity)

                VStack(spacing: 10) {
                    Button {
                        onGotIt()
                    } label: {
                        Text("GOT IT")
                            .font(.headline)
                            .fontWeight(.semibold)
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .background(Theme.brandHeaderStart, in: RoundedRectangle(cornerRadius: 10))

                    Button {
                        onDismissAll()
                    } label: {
                        Text("DISMISS ALL")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundStyle(Theme.secondaryText)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
                }
                .padding(16)
                .background(Theme.appBackground)
            }
            .background(Theme.appBackground)
            .navigationTitle(fundName.isEmpty ? "Fund" : fundName)
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct HoldingDetailView: View {
    let holding: HoldingResponse
    let baseURL: URL
    @ObservedObject var viewModel: AuthViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var healthHistory: [HealthHistoryRecord] = []
    @State private var holdingScoring: DiscoveryScoring?
    @State private var holdingHeadlines: [String] = []
    @State private var sharePricePoints: [StockPriceChartPoint] = []

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                headerCard
                MarketGraphCard(
                    points: sharePricePoints,
                    tradeAt: nil,
                    tradePrice: nil
                )
                secondaryMetaCard
                discoveryCard
                AssessmentAndHealthSectionView(
                    scoring: holdingScoring,
                    healthRecords: healthHistory,
                    headlines: holdingHeadlines,
                    emptyMessage: "No health checks recorded yet."
                )
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarBackButtonHidden(true)
        .task(id: holding.id) {
            async let panel = viewModel.loadHoldingHealthHistory(stockId: holding.stockId)
            async let headlines = viewModel.loadHoldingHeadlines(stockId: holding.stockId)
            async let prices = viewModel.fetchTradeSymbolPriceHistory(symbol: holding.stock.symbol)
            let loaded = await panel
            healthHistory = loaded.history
            holdingScoring = loaded.scoring
            holdingHeadlines = await headlines
            sharePricePoints = await prices
        }
    }

    private var discoveryCard: some View {
        let advisor = normalizedMeta(holding.discoveryName)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                if let logoURL = discoveryLogoURL {
                    AsyncImage(url: logoURL) { image in
                        image.resizable().scaledToFit()
                    } placeholder: {
                        RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
                    }
                    .frame(width: 22, height: 22)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                }
                Text(advisor)
                    .appStyle(.cardTitle)
                    .lineLimit(1)
            }

            Text(DiscoveryExplanationFormatting.attributed(from: holding.discoveryExplanation))
                .detailBody()
                .tint(Theme.link)
                .multilineTextAlignment(.leading)
        }
        .cardSurface()
    }

    private var discoveryLogoURL: URL? {
        guard let logo = holding.discoveryLogo, !logo.isEmpty else { return nil }
        if logo.hasPrefix("http://") || logo.hasPrefix("https://") {
            return URL(string: logo)
        }
        return URL(string: logo, relativeTo: baseURL)?.absoluteURL
    }

    private var headerCard: some View {
        let current = decimal(from: holding.stock.price)
        let average = decimal(from: holding.averagePrice)
        let shares = Decimal(holding.shares)
        let value = (current ?? 0) * shares
        let pnlAmount = (current != nil && average != nil) ? ((current ?? 0) - (average ?? 0)) * shares : nil
        let pnlPercent = computePnlPercent(price: current, average: average)

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.headline)
                        .foregroundStyle(.white)
                }
                .accessibilityLabel("Back")

                VStack(alignment: .leading, spacing: 2) {
                    Text("\(holding.stock.symbol) · \(holding.stock.company ?? holding.stock.symbol)")
                        .appStyle(.screenHeadline)
                        .lineLimit(1)

                    Text(normalizedMeta(holding.stock.industry))
                        .appStyle(.screenSubline)
                        .lineLimit(1)
                }

                Spacer()

                stockLogo(symbol: holding.stock.symbol, size: 24)
            }

            HStack(alignment: .top, spacing: 10) {
                MetricColumn(title: "BUY", value: formatCurrency(average))
                MetricColumn(title: "CURRENT", value: formatCurrency(current))
                MetricColumn(
                    title: "P&L $",
                    value: formatSignedCurrency(pnlAmount),
                    valueColor: Theme.signedColor(for: pnlAmount)
                )
                MetricColumn(
                    title: "P&L %",
                    value: formatPercent(pnlPercent),
                    valueColor: Theme.signedColor(for: pnlPercent)
                )
                Spacer()
            }
            .padding(.top, 10.4)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius))
    }

    private var secondaryMetaCard: some View {
        let current = decimal(from: holding.stock.price)
        let shares = Decimal(holding.shares)
        let value = (current ?? 0) * shares
        return HStack(alignment: .top, spacing: 10) {
            MetricColumn(title: "VALUE", value: formatCurrency(value))
            MetricColumn(
                title: "EXCHANGE",
                value: normalizedMeta(holding.stock.exchange),
                valueColor: Theme.secondaryText
            )
            MetricColumn(
                title: "SECTOR",
                value: normalizedMeta(holding.stock.sector),
                valueColor: Theme.secondaryText
            )
            MetricColumn(title: "SHARES", value: String(holding.shares))
            Spacer()
        }
        .cardSurface()
    }

    private func imageTickerPair(symbol: String) -> some View {
        VStack(spacing: 4) {
            AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
                image.resizable().scaledToFit()
            } placeholder: {
                RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
            }
            .frame(width: 30, height: 30)

            Text(symbol)
                .appStyle(.tickerSymbol)
                .lineLimit(1)
        }
        .frame(width: 54, alignment: .leading)
    }

    private func stockLogo(symbol: String, size: CGFloat) -> some View {
        AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
            image.resizable().scaledToFit()
        } placeholder: {
            RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 5))
    }

    private func normalizedMeta(_ value: String?) -> String {
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "—"
        }
        return value
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

    private func formatSignedCurrency(_ value: Decimal?) -> String {
        guard let value else { return "$0.00" }
        let formatted = formatCurrency(abs(value))
        if value > 0 { return "+\(formatted)" }
        if value < 0 { return "-\(formatted)" }
        return formatted
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "0.00%" }
        return String(format: "%@%.2f%%", value >= 0 ? "+" : "", value)
    }
}

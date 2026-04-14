import SwiftUI

struct TradesView: View {
    @ObservedObject var viewModel: AuthViewModel
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 8) {
                if let fund = viewModel.selectedFund {
                    FundSummaryCard(fund: fund)
                        .padding(.horizontal, 6)
                        .padding(.top, 6)
                }

                List {
                    WealthChartCard(points: viewModel.selectedFundHistory)
                        .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)

                    if viewModel.trades.isEmpty {
                        VStack(spacing: 8) {
                            Text("No trades to show.")
                                .font(.headline)
                                .foregroundStyle(.white)
                            Text("Select a fund to view its trade history.")
                                .font(.footnote)
                                .foregroundStyle(Theme.secondaryText)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 24)
                        .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                    } else {
                        ForEach(viewModel.trades) { trade in
                            Button {
                                path.append(trade.id)
                            } label: {
                                HStack(spacing: 12) {
                                    imageTickerPair(symbol: trade.stock.symbol)
                                    middleCompanySharesPair(trade: trade)
                                    Spacer()
                                    rightAmountActionPair(trade: trade)
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
            .navigationDestination(for: Int.self) { tradeId in
                if let trade = viewModel.trades.first(where: { $0.id == tradeId }) {
                    TradeDetailView(trade: trade, viewModel: viewModel)
                } else {
                    Text("This trade is no longer available.")
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                }
            }
        }
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
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
        }
        .frame(width: 50, alignment: .leading)
    }

    private func middleCompanySharesPair(trade: TradeResponse) -> some View {
        let datePrefix = tradeDatePrefix(from: trade.created)
        return VStack(alignment: .leading, spacing: 3) {
            Text(trade.stock.company ?? trade.stock.symbol)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
                .truncationMode(.tail)

            Text("\(datePrefix) - \(trade.shares) @ \(formatCurrency(decimal(from: trade.price)))")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func rightAmountActionPair(trade: TradeResponse) -> some View {
        let priceDecimal = decimal(from: trade.price) ?? 0
        let total = Decimal(trade.shares) * priceDecimal
        let isBuy = trade.action.uppercased() == "BUY"
        let actionColor: Color = isBuy ? .green : .red
        let amountColor: Color = {
            guard !isBuy else { return Theme.valuePrimary }
            guard let cost = decimal(from: trade.cost), cost != 0 else { return Theme.valuePrimary }
            let pnlPerShare = priceDecimal - cost
            if pnlPerShare > 0 { return .green }
            if pnlPerShare < 0 { return .red }
            return Theme.valuePrimary
        }()

        return VStack(alignment: .trailing, spacing: 2) {
            Text(formatCurrency(total))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(amountColor)
                .lineLimit(1)

            Text(trade.action.uppercased())
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(actionColor)
                .lineLimit(1)
        }
        .frame(minWidth: 78, alignment: .trailing)
    }

    private func decimal(from text: String?) -> Decimal? {
        guard let text, !text.isEmpty else { return nil }
        return Decimal(string: text)
    }

    private func formatCurrency(_ value: Decimal?) -> String {
        guard let value else { return "$0.00" }
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSDecimalNumber(decimal: value)) ?? "$0.00"
    }

    private func tradeDatePrefix(from isoString: String?) -> String {
        guard let isoString else { return "Date" }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let parsedWithFractions = formatter.date(from: isoString)
        if parsedWithFractions == nil {
            formatter.formatOptions = [.withInternetDateTime]
        }
        guard let date = parsedWithFractions ?? formatter.date(from: isoString) else { return "Date" }

        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            return "Today"
        }
        if calendar.isDateInYesterday(date) {
            let weekdayFormatter = DateFormatter()
            weekdayFormatter.dateFormat = "EEEE"
            return weekdayFormatter.string(from: date)
        }

        let now = Date()
        if let days = calendar.dateComponents([.day], from: calendar.startOfDay(for: date), to: calendar.startOfDay(for: now)).day,
           days >= 0, days <= 5 {
            let weekdayFormatter = DateFormatter()
            weekdayFormatter.dateFormat = "EEEE"
            return weekdayFormatter.string(from: date)
        }

        let shortFormatter = DateFormatter()
        shortFormatter.dateFormat = "MMM d"
        return shortFormatter.string(from: date)
    }
}

// MARK: - Trade detail (step 1: header mirrors holding detail)

struct TradeDetailView: View {
    let trade: TradeResponse
    @ObservedObject var viewModel: AuthViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var sharePricePoints: [StockPriceChartPoint] = []

    private var tradeExecutionChartDate: Date? {
        guard let isoString = trade.created, !isoString.isEmpty else { return nil }
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return withFraction.date(from: isoString) ?? plain.date(from: isoString)
    }

    private var tradeExecutionPriceDouble: Double? {
        let s = trade.price
        guard !s.isEmpty, let d = Decimal(string: s) else { return nil }
        return NSDecimalNumber(decimal: d).doubleValue
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                headerCard
                SharePriceChartCard(
                    symbol: trade.stock.symbol,
                    points: sharePricePoints,
                    tradeAt: tradeExecutionChartDate,
                    tradePrice: tradeExecutionPriceDouble
                )
                tradeExplanationSection
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarBackButtonHidden(true)
        .task(id: trade.id) {
            sharePricePoints = await viewModel.fetchTradeSymbolPriceHistory(symbol: trade.stock.symbol)
        }
    }

    private var trimmedTradeExplanation: String? {
        let t = (trade.explanation ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    @ViewBuilder
    private var tradeExplanationSection: some View {
        if let text = trimmedTradeExplanation {
            VStack(alignment: .leading, spacing: 6) {
                Text("Explanation")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                Text(text)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(Theme.valuePrimary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 10)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
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
                    Text("\(trade.stock.symbol) · \(trade.stock.company ?? trade.stock.symbol)")
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .lineLimit(1)

                    Text(formatTradeDateTime(trade.created))
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.secondaryText)
                        .lineLimit(1)
                }

                Spacer()

                tradeActionBadge
            }

            tradeMetricsRow
                .padding(.top, 10.4)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private var tradeMetricsRow: some View {
        HStack(alignment: .top, spacing: 10) {
            ForEach(Array(tradeMetricCells.enumerated()), id: \.offset) { _, cell in
                snapshotMetric(title: cell.title, value: cell.value, valueColor: cell.color)
            }
            Spacer()
        }
    }

    private var tradeMetricCells: [(title: String, value: String, color: Color)] {
        let px = decimal(from: trade.price)
        let current = decimal(from: trade.stock.price)
        let sh = Decimal(trade.shares)
        let avgBuy = decimal(from: trade.cost)

        switch trade.action.uppercased() {
        case "BUY":
            let positionValue = (current ?? px).map { sh * $0 }
            let pnl = pnlPercentBuy(buyPrice: px, currentPrice: current)
            return [
                ("BUY", formatCurrency(px), Theme.valuePrimary),
                ("CURRENT", formatCurrency(current), Theme.valuePrimary),
                ("COST", formatCurrency(positionValue), Theme.valuePrimary),
                ("SHRS", String(trade.shares), Theme.valuePrimary),
                ("P&L %", formatPercent(pnl), percentColor(for: pnl)),
            ]
        case "SELL":
            let pnl = pnlPercentSell(sellPrice: px, avgCost: avgBuy)
            let pnlDollars: Decimal? = {
                guard let px, let avgBuy, avgBuy != 0 else { return nil }
                return (px - avgBuy) * sh
            }()
            let profitLossCell: (String, String, Color) = {
                guard let d = pnlDollars else {
                    return ("P&L $", "—", Theme.valuePrimary)
                }
                if d > 0 { return ("PROFIT", formatSignedCurrency(d), .green) }
                if d < 0 { return ("LOSS", formatSignedCurrency(d), .red) }
                return ("PROFIT", formatSignedCurrency(d), Theme.valuePrimary)
            }()
            return [
                ("BUY", formatCurrency(avgBuy), Theme.valuePrimary),
                ("SELL", formatCurrency(px), Theme.valuePrimary),
                profitLossCell,
                ("SHRS", String(trade.shares), Theme.valuePrimary),
                ("P&L %", formatPercent(pnl), percentColor(for: pnl)),
            ]
        default:
            let mid = px.map { sh * $0 }
            return [
                ("PRICE", formatCurrency(px), Theme.valuePrimary),
                ("CURRENT", formatCurrency(current), Theme.valuePrimary),
                ("VALUE", formatCurrency(mid), Theme.valuePrimary),
                ("SHRS", String(trade.shares), Theme.valuePrimary),
                ("P&L %", "—", Theme.valuePrimary),
            ]
        }
    }

    private func snapshotMetric(title: String, value: String, valueColor: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(valueColor)
                .lineLimit(1)
        }
    }

    private func pnlPercentBuy(buyPrice: Decimal?, currentPrice: Decimal?) -> Double? {
        guard let buyPrice, buyPrice != 0, let currentPrice else { return nil }
        let pct = ((currentPrice / buyPrice) - 1) * 100
        return NSDecimalNumber(decimal: pct).doubleValue
    }

    private func pnlPercentSell(sellPrice: Decimal?, avgCost: Decimal?) -> Double? {
        guard let sellPrice, let avgCost, avgCost != 0 else { return nil }
        let pct = ((sellPrice / avgCost) - 1) * 100
        return NSDecimalNumber(decimal: pct).doubleValue
    }

    private func percentColor(for value: Double?) -> Color {
        guard let value else { return Theme.valuePrimary }
        if value > 0 { return .green }
        if value < 0 { return .red }
        return Theme.valuePrimary
    }

    private func decimal(from text: String?) -> Decimal? {
        guard let text, !text.isEmpty else { return nil }
        return Decimal(string: text)
    }

    private func formatCurrency(_ value: Decimal?) -> String {
        guard let value else { return "—" }
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSDecimalNumber(decimal: value)) ?? "—"
    }

    private func formatSignedCurrency(_ value: Decimal) -> String {
        let formatted = formatCurrency(abs(value))
        if value > 0 { return "+\(formatted)" }
        if value < 0 { return "-\(formatted)" }
        return formatted
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%@%.2f%%", value >= 0 ? "+" : "", value)
    }

    private var tradeActionBadge: some View {
        let action = trade.action.uppercased()
        let isBuy = action == "BUY"
        let isSell = action == "SELL"
        let color: Color = {
            if isBuy { return .green }
            if isSell { return .red }
            return Theme.valuePrimary
        }()
        return Text(action)
            .font(.title3)
            .fontWeight(.heavy)
            .foregroundStyle(color)
            .lineLimit(1)
            .minimumScaleFactor(0.75)
    }

    private func formatTradeDateTime(_ isoString: String?) -> String {
        guard let isoString, !isoString.isEmpty else { return "—" }
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let date = withFraction.date(from: isoString) ?? plain.date(from: isoString)
        guard let date else { return isoString }

        let out = DateFormatter()
        out.dateStyle = .medium
        out.timeStyle = .short
        return out.string(from: date)
    }
}

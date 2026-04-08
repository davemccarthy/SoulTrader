import SwiftUI

struct TradesView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        Group {
            if viewModel.trades.isEmpty {
                VStack(spacing: 8) {
                    Text("No trades to show.")
                        .font(.headline)
                        .foregroundStyle(.white)
                    Text("Select a fund to view its trade history.")
                        .font(.footnote)
                        .foregroundStyle(Theme.secondaryText)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
                .padding()
            } else {
                List(viewModel.trades) { trade in
                    HStack(spacing: 12) {
                        imageTickerPair(symbol: trade.stock.symbol)
                        middleCompanySharesPair(trade: trade)
                        Spacer()
                        rightAmountActionPair(trade: trade)
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, 6)
                    .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
                    .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                    .listRowBackground(Color.clear)
                }
                .scrollContentBackground(.hidden)
                .scrollIndicators(.hidden)
                .background(Theme.appBackground)
            }
        }
        .background(Theme.appBackground)
        .toolbar(.hidden, for: .navigationBar)
    }

    private func imageTickerPair(symbol: String) -> some View {
        VStack(spacing: 4) {
            AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
                image.resizable().scaledToFit()
            } placeholder: {
                RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.15))
            }
            .frame(width: 34, height: 34)

            Text(symbol)
                .font(.caption)
                .fontWeight(.bold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
        }
        .frame(width: 56, alignment: .leading)
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

            Text("\(datePrefix) - \(trade.shares) shares @ \(formatCurrency(decimal(from: trade.price)))")
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
        let signedTotal = "\(isBuy ? "-" : "+")\(formatCurrency(total))"
        let actionColor: Color = isBuy ? .green : .red

        return VStack(alignment: .trailing, spacing: 2) {
            Text(signedTotal)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
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
            return "Yesterday"
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

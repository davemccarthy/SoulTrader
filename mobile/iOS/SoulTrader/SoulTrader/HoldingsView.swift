import SwiftUI

struct HoldingsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        VStack(spacing: 8) {
            if let fund = viewModel.selectedFund {
                FundSummaryCard(fund: fund)
                    .padding(.horizontal, 6)
                    .padding(.top, 6)
            }

            Group {
                if viewModel.holdings.isEmpty {
                    VStack(spacing: 8) {
                        Text("No holdings to show.")
                            .font(.headline)
                            .foregroundStyle(.white)
                        Text("Select a fund with holdings on the FUNDS tab.")
                            .font(.footnote)
                            .foregroundStyle(Theme.secondaryText)
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
                        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
                        .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                        .listRowBackground(Color.clear)
                    }
                    .scrollContentBackground(.hidden)
                    .scrollIndicators(.hidden)
                    .background(Theme.appBackground)
                }
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

    private func middleCompanyInvestmentPair(holding: HoldingResponse) -> some View {
        let avgDecimal = decimal(from: holding.averagePrice) ?? 0
        let sharesDecimal = Decimal(holding.shares)
        let investment = sharesDecimal * avgDecimal

        return VStack(alignment: .leading, spacing: 3) {
            Text(holding.stock.company ?? holding.stock.symbol)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
                .truncationMode(.tail)

            Text(formatCurrency(investment))
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func pricePnlPair(holding: HoldingResponse) -> some View {
        let priceDecimal = decimal(from: holding.stock.price)
        let avgDecimal = decimal(from: holding.averagePrice)
        let pnlPercent = computePnlPercent(price: priceDecimal, average: avgDecimal)
        let pnlColor: Color = {
            guard let pnlPercent else { return Theme.valuePrimary }
            if pnlPercent > 0 { return .green }
            if pnlPercent < 0 { return .red }
            return Theme.valuePrimary
        }()

        return VStack(alignment: .trailing, spacing: 2) {
            Text(formatCurrency(priceDecimal))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
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

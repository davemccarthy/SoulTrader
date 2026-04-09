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

            List {
                WealthChartCard(points: viewModel.selectedFundHistory)
                    .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)

                if viewModel.holdings.isEmpty {
                    VStack(spacing: 8) {
                        Text("No holdings to show.")
                            .font(.headline)
                            .foregroundStyle(.white)
                        Text("Select a fund with holdings on the FUNDS tab.")
                            .font(.footnote)
                            .foregroundStyle(Theme.secondaryText)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 24)
                    .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                } else {
                    ForEach(viewModel.holdings) { holding in
                        NavigationLink(destination: HoldingDetailView(holding: holding)) {
                            HStack(spacing: 12) {
                                imageTickerPair(symbol: holding.stock.symbol)
                                middleCompanyInvestmentPair(holding: holding)
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
            .contentMargins(.top, 0, for: .scrollContent)
            .background(Theme.appBackground)
        }
        .background(Theme.appBackground)
        .toolbar(.hidden, for: .navigationBar)
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

struct HoldingDetailView: View {
    let holding: HoldingResponse
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                headerCard
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarBackButtonHidden(true)
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

                Text("\(holding.stock.symbol) · \(holding.stock.company ?? holding.stock.symbol)")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
                    .lineLimit(1)

                Spacer()

                stockLogo(symbol: holding.stock.symbol, size: 24)
            }

            HStack(spacing: 12) {
                middleCompanyIndustryPair(holding: holding)
                Spacer()
                pricePnlPair(holding: holding)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
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
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(Theme.valuePrimary)
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

    private func middleCompanyIndustryPair(holding: HoldingResponse) -> some View {
        let industry = normalizedIndustry(holding.stock.industry)
        return VStack(alignment: .leading, spacing: 3) {
            Text("Industry")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)

            Text(industry)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.secondaryText)
                .lineLimit(1)
                .truncationMode(.tail)
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
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
            Text(formatPercent(pnlPercent))
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(pnlColor)
                .lineLimit(1)
        }
        .frame(minWidth: 68, alignment: .trailing)
    }

    private func normalizedIndustry(_ industry: String?) -> String {
        guard let industry, !industry.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return "Industry unavailable"
        }
        return industry
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

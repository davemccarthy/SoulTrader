import SwiftUI

struct FundsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        VStack(spacing: 8) {
            if let dashboard = viewModel.globalDashboard {
                GlobalSummaryCard(dashboard: dashboard)
                    .padding(.horizontal, 6)
                    .padding(.top, 6)
            }

            List {
                WealthChartCard(points: viewModel.globalHistory)
                    .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)

                if let dashboard = viewModel.globalDashboard {
                    FundSecondarySummaryCard(
                        countTitle: "FUNDS",
                        countValue: String(viewModel.funds.count),
                        equityPercent: equityPercent(
                            totalValue: dashboard.totalValue,
                            portfolioValue: dashboard.holdingsMarketValue
                        ),
                        middleTitle: "RETURN",
                        middleValue: formatCurrency(dashboard.returnAmount),
                        middleColor: amountColor(dashboard.returnAmount),
                        todayPercent: dashboard.todayPercent
                    )
                    .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }

                ForEach(viewModel.funds) { fund in
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
                            value: formatCurrency(fund.dashboard.holdingsMarketValue),
                            color: percentColor(fund.dashboard.holdingsPnl),
                            alignment: .trailing
                        )

                        metricPair(
                            title: "P&L",
                            value: formatPercent(fund.dashboard.returnPercent),
                            color: percentColor(fund.dashboard.returnPercent),
                            alignment: .trailing
                        )
                        }

                        HStack(spacing: 10) {
                        Text("\(fund.dashboard.holdingsCount) stocks")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(Theme.valuePrimary)
                        Spacer()
                        miniPair(
                            title: "ABV:",
                            value: formatPercent(fund.dashboard.estAbvPercent),
                            color: percentColor(fund.dashboard.estAbvPercent)
                        )
                        miniPair(
                            title: "TODAY:",
                            value: formatPercent(fund.dashboard.todayPercent),
                            color: percentColor(fund.dashboard.todayPercent)
                        )
                        }
                    }
                    .padding(.vertical, 4)
                    .padding(.horizontal, 6)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        Task { await viewModel.selectFund(fund.id) }
                    }
                    .background(
                        viewModel.selectedFundId == fund.id
                            ? Color.green.opacity(0.08)
                            : Theme.rowBackground,
                        in: RoundedRectangle(cornerRadius: 10)
                    )
                    .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
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
    }

    private func metricPair(
        title: String,
        value: String,
        color: Color = Theme.valuePrimary,
        alignment: Alignment,
        isFlexible: Bool = true
    ) -> some View {
        VStack(alignment: alignment == .leading ? .leading : .trailing, spacing: 2) {
            Text(title)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
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
        color: Color = Theme.valuePrimary
    ) -> some View {
        HStack(spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(Theme.labelAccent)
            Text(value)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(color)
        }
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
        let normalized = normalizedPercent(value)
        return String(format: "%.2f%%", normalized)
    }

    private func percentColor(_ value: Double) -> Color {
        let normalized = normalizedPercent(value)
        if normalized > 0 { return .green }
        if normalized < 0 { return .red }
        return Theme.valuePrimary
    }

    private func normalizedPercent(_ value: Double) -> Double {
        abs(value) < 0.005 ? 0.0 : value
    }

    private func amountColor(_ value: Double) -> Color {
        if value > 0 { return .green }
        if value < 0 { return .red }
        return Theme.valuePrimary
    }

    private func equityPercent(totalValue: Double, portfolioValue: Double) -> Double? {
        guard totalValue > 0 else { return nil }
        return (portfolioValue / totalValue) * 100
    }
}

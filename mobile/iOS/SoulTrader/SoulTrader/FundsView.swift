import SwiftUI

struct FundsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        VStack(spacing: 8) {
            if let dashboard = viewModel.globalDashboard {
                GlobalSummaryCard(
                    dashboard: dashboard,
                    totalPercentTitle: viewModel.totalPercentTitle,
                    totalPercentValue: viewModel.totalPercentValue(for: dashboard),
                    onTap: {
                        viewModel.toggleReturnPercentMode()
                    }
                )
                    .padding(.horizontal, 6)
                    .padding(.top, 6)
            }

            List {
                if let dashboard = viewModel.globalDashboard {
                    FundSecondarySummaryCard(
                        countTitle: "FUNDS",
                        countValue: String(viewModel.funds.count),
                        equityPercent: equityPercent(
                            totalValue: dashboard.totalValue,
                            portfolioValue: dashboard.holdingsMarketValue
                        ),
                        middleTitle: "RETURN",
                        middleValue: Theme.formatCompactCurrency(dashboard.returnAmount),
                        middleColor: Theme.signedColor(for: dashboard.returnAmount),
                        todayPercent: dashboard.todayPercent
                    )
                    .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                    .listRowBackground(Color.clear)
                    .listRowSeparator(.hidden)
                }

                ForEach(viewModel.funds) { fund in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(alignment: .top, spacing: 0) {
                        MetricColumn(
                            title: "FUND",
                            value: fund.name.isEmpty ? "Unnamed" : fund.name,
                            expands: false
                        )
                        .frame(width: 64, alignment: .leading)

                        MetricColumn(
                            title: "PORTFOLIO",
                            value: Theme.formatCompactCurrency(fund.dashboard.totalValue)
                        )
                        .frame(width: 128, alignment: .leading)

                        MetricColumn(
                            title: "HOLDINGS",
                            value: Theme.formatCompactCurrency(fund.dashboard.holdingsMarketValue),
                            valueColor: Theme.signedColor(for: fund.dashboard.holdingsPnl),
                            alignment: .trailing,
                            expands: true
                        )

                        MetricColumn(
                            title: viewModel.totalPercentTitle,
                            value: formatPercent(viewModel.totalPercentValue(for: fund.dashboard)),
                            valueColor: Theme.signedColor(for: viewModel.totalPercentValue(for: fund.dashboard)),
                            alignment: .trailing,
                            expands: true
                        )
                        }

                        HStack(spacing: 10) {
                        Text("\(fund.dashboard.holdingsCount) stocks")
                            .appStyle(.inlineMetricValue)
                        Spacer()
                        InlineMetricPair(
                            title: "APR:",
                            value: formatPercent(fund.dashboard.estAprPercent),
                            valueColor: Theme.signedColor(for: fund.dashboard.estAprPercent)
                        )
                        InlineMetricPair(
                            title: "TODAY:",
                            value: formatPercent(fund.dashboard.todayPercent),
                            valueColor: Theme.signedColor(for: fund.dashboard.todayPercent)
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
                        in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius)
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

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        let normalized = normalizedPercent(value)
        return String(format: "%.2f%%", normalized)
    }

    private func normalizedPercent(_ value: Double) -> Double {
        abs(value) < 0.005 ? 0.0 : value
    }

    private func equityPercent(totalValue: Double, portfolioValue: Double) -> Double? {
        guard totalValue > 0 else { return nil }
        return (portfolioValue / totalValue) * 100
    }
}

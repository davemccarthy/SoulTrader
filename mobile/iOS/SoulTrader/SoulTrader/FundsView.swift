import SwiftUI

struct FundsView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        List(viewModel.funds) { fund in
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
                        value: formatPercent(fund.dashboard.holdingsPnl),
                        color: fund.dashboard.holdingsPnl >= 0 ? .green : .red,
                        alignment: .trailing
                    )

                    metricPair(
                        title: "P&L",
                        value: formatPercent(fund.dashboard.returnPercent),
                        color: fund.dashboard.returnPercent >= 0 ? .green : .red,
                        alignment: .trailing
                    )
                }

                HStack(spacing: 10) {
                    miniPair(title: "DUR:", value: "\(fund.dashboard.estabDays) days")
                    Spacer()
                    miniPair(
                        title: "EST ABV:",
                        value: formatPercent(fund.dashboard.estAbvPercent),
                        color: fund.dashboard.estAbvPercent >= 0 ? .green : .red
                    )
                    miniPair(title: "TODAY:", value: "0.00%")
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
            .onTapGesture {
                Task { await viewModel.selectFund(fund.id) }
            }
            .background(
                viewModel.selectedFundId == fund.id
                    ? Color.green.opacity(0.08)
                    : Color.clear
            )
            .listRowBackground(Theme.rowBackground)
            .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
        }
        .scrollContentBackground(.hidden)
        .scrollIndicators(.hidden)
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
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSNumber(value: value)) ?? "$0.00"
    }

    private func formatPercent(_ value: Double) -> String {
        String(format: "%.2f%%", value)
    }
}

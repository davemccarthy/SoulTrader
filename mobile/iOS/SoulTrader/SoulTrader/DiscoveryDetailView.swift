import SwiftUI

struct DiscoveryDetailView: View {
    let discoveryId: Int
    let baseURL: URL
    @ObservedObject var viewModel: AuthViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var detail: DiscoveryDetailResponse?
    @State private var loadError: String?
    @State private var sharePricePoints: [StockPriceChartPoint] = []

    var body: some View {
        Group {
            if let loadError, detail == nil {
                Text(loadError)
                    .font(.footnote)
                    .foregroundStyle(Theme.secondaryText)
                    .padding()
            } else if let detail {
                discoveryContent(detail)
            } else {
                ProgressView()
                    .tint(.white)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarBackButtonHidden(true)
        .task(id: discoveryId) { await load() }
    }

    private func discoveryContent(_ detail: DiscoveryDetailResponse) -> some View {
        ScrollView {
            VStack(spacing: 10) {
                headerCard(detail)
                MarketGraphCard(
                    points: sharePricePoints,
                    tradeAt: nil,
                    tradePrice: nil
                )
                secondaryMetaCard(detail)
                explanationCard(detail)
                healthSection(detail)
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
    }

    private func healthSection(_ detail: DiscoveryDetailResponse) -> some View {
        Group {
            if let h = detail.health {
                HealthHistoryRecordCard(record: h, checkNumber: nil)
            } else {
                Text("No health check recorded for this discovery.")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.secondaryText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                    .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func headerCard(_ detail: DiscoveryDetailResponse) -> some View {
        let stock = detail.stock
        let current = decimal(from: stock.price)
        let disc = decimal(from: detail.discoveryPrice)
        let chg = percentChange(from: disc, to: current)

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
                    Text("\(stock.symbol) · \(stock.company ?? stock.symbol)")
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .lineLimit(1)

                    Text(normalizedMeta(stock.industry))
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.secondaryText)
                        .lineLimit(1)
                }

                Spacer()

                stockLogo(symbol: stock.symbol, size: 24)
            }

            HStack(alignment: .top, spacing: 10) {
                snapshotMetric(
                    title: "CURRENT",
                    value: formatCurrency(current),
                    valueColor: Theme.valuePrimary
                )
                snapshotMetric(
                    title: "CHG %",
                    value: formatPercent(chg),
                    valueColor: percentColor(for: chg)
                )
                snapshotMetric(
                    title: "AT DISC",
                    value: formatCurrency(disc),
                    valueColor: Theme.valuePrimary
                )
                snapshotMetric(
                    title: "SCORE",
                    value: formatOptionalScore(detail.health?.score),
                    valueColor: Theme.valuePrimary
                )
                Spacer()
            }
            .padding(.top, 10.4)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func explanationCard(_ detail: DiscoveryDetailResponse) -> some View {
        let advisor = normalizedMeta(detail.advisor.name)
        let explTrim = detail.explanation.trimmingCharacters(in: .whitespacesAndNewlines)
        let explRaw: String? = explTrim.isEmpty ? nil : detail.explanation
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center, spacing: 8) {
                if let logoURL = advisorLogoURL(detail.advisor.logoUrl) {
                    AsyncImage(url: logoURL) { image in
                        image.resizable().scaledToFit()
                    } placeholder: {
                        RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
                    }
                    .frame(width: 22, height: 22)
                    .clipShape(RoundedRectangle(cornerRadius: 5))
                }
                Text(advisor)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.valuePrimary)
                    .lineLimit(1)
            }

            Text(DiscoveryExplanationFormatting.attributed(from: explRaw))
                .font(.subheadline)
                .fontWeight(.light)
                .foregroundStyle(Theme.valuePrimary.opacity(0.95))
                .tint(Color(red: 0.45, green: 0.78, blue: 1.0))
                .multilineTextAlignment(.leading)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func advisorLogoURL(_ logo: String?) -> URL? {
        guard let logo, !logo.isEmpty else { return nil }
        if logo.hasPrefix("http://") || logo.hasPrefix("https://") {
            return URL(string: logo)
        }
        return URL(string: logo, relativeTo: baseURL)?.absoluteURL
    }

    private func secondaryMetaCard(_ detail: DiscoveryDetailResponse) -> some View {
        let stock = detail.stock
        return HStack(alignment: .top, spacing: 10) {
            snapshotMetric(
                title: "DISCOVERED",
                value: formatShortDate(detail.created),
                valueColor: Theme.secondaryText
            )
            snapshotMetric(
                title: "EXCHANGE",
                value: normalizedMeta(stock.exchange),
                valueColor: Theme.secondaryText
            )
            snapshotMetric(
                title: "SECTOR",
                value: normalizedMeta(stock.sector),
                valueColor: Theme.secondaryText
            )
            snapshotMetric(
                title: "INDUSTRY",
                value: normalizedMeta(stock.industry),
                valueColor: Theme.secondaryText
            )
            Spacer()
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
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

    private func percentChange(from base: Decimal?, to price: Decimal?) -> Double? {
        guard let base, let price, base != 0 else { return nil }
        let percent = ((price / base) - 1) * 100
        return NSDecimalNumber(decimal: percent).doubleValue
    }

    private func formatCurrency(_ value: Decimal?) -> String {
        guard let value else { return "—" }
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        formatter.maximumFractionDigits = 2
        return formatter.string(from: NSDecimalNumber(decimal: value)) ?? "—"
    }

    private func formatPercent(_ value: Double?) -> String {
        guard let value else { return "—" }
        return String(format: "%.2f%%", value)
    }

    private func formatOptionalScore(_ v: Double?) -> String {
        guard let v else { return "—" }
        if abs(v) < 1e-9 {
            return "AVOID"
        }
        return String(format: "%.1f", v)
    }

    private func formatShortDate(_ iso: String?) -> String {
        guard let iso, !iso.isEmpty else { return "—" }
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let date = withFraction.date(from: iso) ?? plain.date(from: iso)
        guard let date else { return "—" }
        let out = DateFormatter()
        out.dateStyle = .medium
        out.timeStyle = .short
        return out.string(from: date)
    }

    private func percentColor(for value: Double?) -> Color {
        guard let value else { return Theme.valuePrimary }
        if value > 0 { return .green }
        if value < 0 { return .red }
        return Theme.valuePrimary
    }

    private func load() async {
        detail = nil
        loadError = nil
        sharePricePoints = []
        do {
            let d = try await viewModel.fetchDiscoveryDetail(discoveryId: discoveryId)
            detail = d
            sharePricePoints = await viewModel.fetchTradeSymbolPriceHistory(symbol: d.stock.symbol)
        } catch {
            loadError = error.localizedDescription
        }
    }
}

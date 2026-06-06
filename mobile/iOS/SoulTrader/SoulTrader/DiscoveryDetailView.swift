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
                    .appStyle(.emptyStateMessage)
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
                AssessmentAndHealthSectionView(
                    scoring: detail.scoring,
                    healthRecords: detail.health.map { [$0] } ?? [],
                    emptyMessage: "No assessment recorded for this discovery."
                )
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
    }

    private func headerScoreText(for detail: DiscoveryDetailResponse) -> String {
        if let scoring = detail.scoring {
            let text = scoring.displayScoreText
            if text != "—" { return text }
        }
        return formatOptionalScore(detail.health?.score)
    }

    private func headerCard(_ detail: DiscoveryDetailResponse) -> some View {
        let stock = detail.stock
        let current = decimal(from: stock.price)
        let disc = decimal(from: detail.discoveryPrice)
        let chg = percentChange(from: disc, to: current)
        let discoveredLine: String = {
            let when = formatShortDate(detail.created)
            if when == "—" { return "Discovered —" }
            return "Discovered \(when)"
        }()

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
                        .appStyle(.screenHeadline)
                        .lineLimit(1)

                    Text(discoveredLine)
                        .appStyle(.screenSubline)
                        .lineLimit(1)
                }

                Spacer()

                stockLogo(symbol: stock.symbol, size: 24)
            }

            HStack(alignment: .top, spacing: 10) {
                MetricColumn(title: "CURRENT", value: formatCurrency(current))
                MetricColumn(
                    title: "CHG %",
                    value: formatPercent(chg),
                    valueColor: Theme.signedColor(for: chg)
                )
                MetricColumn(title: "AT DISC", value: formatCurrency(disc))
                MetricColumn(title: "GRADE", value: headerScoreText(for: detail))
                Spacer()
            }
            .padding(.top, 10.4)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius))
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
                    .appStyle(.cardTitle)
                    .lineLimit(1)
            }

            Text(DiscoveryExplanationFormatting.attributed(from: explRaw))
                .detailBody()
                .tint(Theme.link)
                .multilineTextAlignment(.leading)
        }
        .cardSurface()
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
            MetricColumn(
                title: "EXCHANGE",
                value: normalizedMeta(stock.exchange),
                valueColor: Theme.secondaryText
            )
            MetricColumn(
                title: "SECTOR",
                value: normalizedMeta(stock.sector),
                valueColor: Theme.secondaryText
            )
            MetricColumn(
                title: "INDUSTRY",
                value: normalizedMeta(stock.industry),
                valueColor: Theme.secondaryText
            )
            Spacer()
        }
        .cardSurface()
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

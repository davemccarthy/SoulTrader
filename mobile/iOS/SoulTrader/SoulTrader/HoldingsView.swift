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
                        NavigationLink(destination: HoldingDetailView(holding: holding, baseURL: viewModel.selectedHost.baseURL, viewModel: viewModel)) {
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
    let baseURL: URL
    @ObservedObject var viewModel: AuthViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var healthHistory: [HealthHistoryRecord] = []

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                headerCard
                discoveryCard
                secondaryMetaCard
                healthSection
            }
            .padding(.horizontal, 6)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarBackButtonHidden(true)
        .task {
            healthHistory = await viewModel.loadHoldingHealthHistory(stockId: holding.stockId)
        }
    }

    private var healthSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            if healthHistory.isEmpty {
                Text("No health checks recorded yet.")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.secondaryText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                    .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
            } else {
                ForEach(healthHistory) { record in
                    healthRecordCard(record)
                }
            }
        }
    }

    private func healthRecordCard(_ record: HealthHistoryRecord) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("Health check")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                Spacer()
                Text(String(format: "Score %.1f", record.score))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.valuePrimary)
            }
            
            Text(formatHealthDate(record.created))
                .font(.subheadline)
                .fontWeight(.light)
                .foregroundStyle(Theme.secondaryText)

            if record.renderKind == "edgar" {
                Text("EDGAR Ex-99")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                Text(edgarHealthSummary(record))
                    .font(.subheadline)
                    .fontWeight(.light)
                    .foregroundStyle(Theme.valuePrimary.opacity(0.92))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Algorithm components")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                healthMetricRow(label: "Confidence", value: record.confidenceScore?.display)
                healthMetricRow(label: "Health", value: record.healthScore?.display)
                healthMetricRow(label: "Valuation", value: record.valuationScore?.display)
                healthMetricRow(label: "Piotroski", value: piotroskiDisplay(record.piotroski))
                healthMetricRow(label: "Altman Z", value: record.altmanZ?.display)

                if record.overlayPoints != nil || !record.overlayReasons.isEmpty {
                    Text("Overlay")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.labelAccent)
                        .padding(.top, 2)
                    healthMetricRow(label: "Points", value: overlayPointsLabel(record.overlayPoints))
                    if !record.overlayReasons.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(record.overlayReasons.enumerated()), id: \.offset) { _, reason in
                                Text("• \(reason)")
                                    .font(.subheadline)
                                    .fontWeight(.light)
                                    .foregroundStyle(Theme.secondaryText)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }

                if record.geminiWeight != nil || record.geminiRec != nil || (record.geminiExplanation.map { $0.display != "—" } ?? false) {
                    Text("Gemini")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.labelAccent)
                        .padding(.top, 2)
                    healthMetricRow(label: "Weight", value: record.geminiWeight?.display)
                    healthMetricRow(label: "Recommendation", value: record.geminiRec?.display)
                    if let gem = record.geminiExplanation?.display, gem != "—" {
                        Text(gem)
                            .font(.subheadline)
                            .fontWeight(.light)
                            .foregroundStyle(Theme.valuePrimary.opacity(0.95))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func edgarHealthSummary(_ record: HealthHistoryRecord) -> String {
        let summary = record.meta?.media?.summary?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if summary.isEmpty {
            return "No media summary in this health record."
        }
        return summary
    }

    private func healthMetricRow(label: String, value: String?) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
            Spacer()
            Text(value ?? "—")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .multilineTextAlignment(.trailing)
        }
    }

    private func piotroskiDisplay(_ scalar: HealthScalar?) -> String? {
        guard let s = scalar?.display, s != "—" else { return nil }
        if s.contains("/") { return s }
        if Int(s) != nil { return "\(s)/4" }
        return s
    }

    private func overlayPointsLabel(_ pts: Double?) -> String? {
        guard let pts else { return nil }
        if pts >= 0 {
            return String(format: "+%.1f", pts)
        }
        return String(format: "%.1f", pts)
    }

    private func formatHealthDate(_ iso: String?) -> String {
        guard let iso, !iso.isEmpty else { return "—" }
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let date = withFraction.date(from: iso) ?? plain.date(from: iso)
        guard let date else {
            return iso
        }
        let out = DateFormatter()
        out.dateStyle = .medium
        out.timeStyle = .short
        return out.string(from: date)
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
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.valuePrimary)
                    .lineLimit(1)
            }

            Text(discoveryExplanationAttributed(from: holding.discoveryExplanation))
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
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .lineLimit(1)

                    Text(normalizedMeta(holding.stock.industry))
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(Theme.secondaryText)
                        .lineLimit(1)
                }

                Spacer()

                stockLogo(symbol: holding.stock.symbol, size: 24)
            }

            HStack(alignment: .top, spacing: 10) {
                snapshotMetric(
                    title: "CURRENT",
                    value: formatCurrency(current),
                    valueColor: Theme.valuePrimary
                )
                snapshotMetric(
                    title: "P&L %",
                    value: formatPercent(pnlPercent),
                    valueColor: percentColor(for: pnlPercent)
                )
                snapshotMetric(
                    title: "P&L $",
                    value: formatSignedCurrency(pnlAmount),
                    valueColor: amountColor(for: pnlAmount)
                )
                snapshotMetric(
                    title: "VALUE",
                    value: formatCurrency(value),
                    valueColor: Theme.valuePrimary
                )
                Spacer()
            }
            .padding(.top, 8)
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private var secondaryMetaCard: some View {
        let average = decimal(from: holding.averagePrice)
        return HStack(alignment: .top, spacing: 10) {
            snapshotMetric(
                title: "AVG",
                value: formatCurrency(average),
                valueColor: Theme.valuePrimary
            )
            snapshotMetric(
                title: "EXCHANGE",
                value: normalizedMeta(holding.stock.exchange),
                valueColor: Theme.secondaryText
            )
            snapshotMetric(
                title: "SECTOR",
                value: normalizedMeta(holding.stock.sector),
                valueColor: Theme.secondaryText
            )
            snapshotMetric(
                title: "SHARES",
                value: String(holding.shares),
                valueColor: Theme.valuePrimary
            )
            Spacer()
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
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

    /// `|` segments; `Article: title` + next `https?://...` → link; bare URL → link. Plain segments separated by newline; any piece next to a link uses a blank line so links read as standalone paragraphs.
    private func discoveryExplanationAttributed(from raw: String?) -> AttributedString {
        let pieces = parseDiscoveryExplanationPieces(raw)
        guard !pieces.isEmpty else {
            return AttributedString("No discovery explanation available.")
        }
        var result = AttributedString()
        for (idx, piece) in pieces.enumerated() {
            if idx > 0 {
                let prev = pieces[idx - 1]
                let paragraphBreak = discoveryExplanationPieceIsLink(prev) || discoveryExplanationPieceIsLink(piece)
                result.append(AttributedString(paragraphBreak ? "\n\n" : "\n"))
            }
            switch piece {
            case .plain(let s):
                result.append(AttributedString(s))
            case .articleLink(let title, let url):
                var linkText = AttributedString(title)
                linkText.link = url
                result.append(linkText)
            case .bareURL(let url):
                var linkText = AttributedString(url.absoluteString)
                linkText.link = url
                result.append(linkText)
            }
        }
        return result
    }

    private func discoveryExplanationPieceIsLink(_ piece: DiscoveryExplanationPiece) -> Bool {
        switch piece {
        case .plain: return false
        case .articleLink, .bareURL: return true
        }
    }

    private enum DiscoveryExplanationPiece {
        case plain(String)
        case articleLink(title: String, url: URL)
        case bareURL(URL)
    }

    private func parseDiscoveryExplanationPieces(_ raw: String?) -> [DiscoveryExplanationPiece] {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        let normalized = raw.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return [] }
        let segments = normalized.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var pieces: [DiscoveryExplanationPiece] = []
        var i = 0
        while i < segments.count {
            let current = segments[i]
            if let title = discoveryArticleTitle(from: current), i + 1 < segments.count {
                let urlRaw = segments[i + 1]
                if urlRaw.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil,
                   let url = URL(string: urlRaw) {
                    pieces.append(.articleLink(title: title, url: url))
                    i += 2
                    continue
                }
            }
            if current.range(of: #"^https?://\S+$"#, options: .regularExpression) != nil,
               let url = URL(string: current) {
                pieces.append(.bareURL(url))
                i += 1
                continue
            }
            pieces.append(.plain(current))
            i += 1
        }
        return pieces
    }

    private func discoveryArticleTitle(from segment: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: "^article\\s*:\\s*(.+)$", options: .caseInsensitive) else { return nil }
        let range = NSRange(segment.startIndex..., in: segment)
        guard let match = regex.firstMatch(in: segment, range: range),
              let titleRange = Range(match.range(at: 1), in: segment) else { return nil }
        let title = String(segment[titleRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        return title.isEmpty ? nil : title
    }

    private func percentColor(for value: Double?) -> Color {
        guard let value else { return Theme.valuePrimary }
        if value > 0 { return .green }
        if value < 0 { return .red }
        return Theme.valuePrimary
    }

    private func amountColor(for value: Decimal?) -> Color {
        guard let value else { return Theme.valuePrimary }
        if value > 0 { return .green }
        if value < 0 { return .red }
        return Theme.valuePrimary
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

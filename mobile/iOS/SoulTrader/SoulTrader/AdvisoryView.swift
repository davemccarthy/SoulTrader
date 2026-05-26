import SwiftUI

struct AdvisoryView: View {
    /// Tap-to-cycle values for advisory lookback.
    private let lookbackOptions = [7, 30, 90]

    @ObservedObject var viewModel: AuthViewModel
    @State private var path = NavigationPath()
    @State private var advisors: [FundAdvisorRow] = []
    @State private var statsByAdvisorId: [Int: AdvisorScoreboardRow] = [:]
    @State private var selectedLookbackDays = 30
    @State private var loadError: String?
    @State private var isLoading = false

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if isLoading && advisors.isEmpty && loadError == nil {
                    ProgressView()
                        .tint(.white)
                } else if let loadError {
                    Text(loadError)
                        .appStyle(.emptyStateMessage)
                        .padding()
                } else if advisors.isEmpty {
                    Text("No advisors for this fund.")
                        .appStyle(.emptyStateMessage)
                        .padding()
                } else {
                    List {
                        AdvisoryTopSummaryCard(
                            advisorCount: advisors.count,
                            winners: statsByAdvisorId.values.reduce(0) { $0 + $1.winners },
                            losers: statsByAdvisorId.values.reduce(0) { $0 + $1.losers },
                            lookbackDays: selectedLookbackDays,
                            onTapLookback: cycleLookback
                        )
                        .listRowInsets(EdgeInsets(top: 0, leading: 6, bottom: 8, trailing: 6))
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)

                        ForEach(advisors) { row in
                            Button {
                                path.append(
                                    AdvisorNav(
                                        id: row.id,
                                        name: row.name,
                                        description: row.description,
                                        imageUrl: row.imageUrl
                                    )
                                )
                            } label: {
                                advisorRow(row, stats: statsByAdvisorId[row.id])
                            }
                            .buttonStyle(.plain)
                            .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.appBackground)
            .navigationDestination(for: AdvisorNav.self) { nav in
                AdvisorDiscoveriesView(
                    viewModel: viewModel,
                    navigationPath: $path,
                    advisorId: nav.id,
                    advisorName: nav.name,
                    advisorDescription: nav.description,
                    advisorImageUrl: nav.imageUrl,
                    advisorStats: statsByAdvisorId[nav.id],
                    lookbackDays: selectedLookbackDays
                )
            }
            .navigationDestination(for: DiscoveryDetailNav.self) { nav in
                DiscoveryDetailView(
                    discoveryId: nav.discoveryId,
                    baseURL: viewModel.apiBaseURL,
                    viewModel: viewModel
                )
            }
        }
        .task(id: viewModel.selectedFundId) { await load() }
        .onChange(of: viewModel.selectedFundId) { _, _ in
            path = NavigationPath()
        }
        .onChange(of: selectedLookbackDays) { _, _ in
            Task { await load() }
        }
        .onChange(of: viewModel.selectedTab) { _, tab in
            if tab == .advisory { Task { await load() } }
        }
    }

    private func advisorRow(_ row: FundAdvisorRow, stats: AdvisorScoreboardRow?) -> some View {
        HStack(alignment: .top, spacing: 12) {
            advisorImage(urlString: row.imageUrl)
                .frame(width: 44, height: 44)
            VStack(alignment: .leading, spacing: 4) {
                Text(row.name)
                    .appStyle(.listHeadline)
                if !row.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(row.description.trimmingCharacters(in: .whitespacesAndNewlines))
                        .appStyle(.listSubline)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let s = stats {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("\(s.trades) \(s.trades == 1 ? "buy" : "buys") · \(String(format: "%.0f%%", s.winRate)) win")
                            .appStyle(.metricLabel)
                        AdvisorWinRateBar(winRate: s.winRate)
                            .frame(maxWidth: 200)
                    }
                    .padding(.top, 2)
                }
            }
            Spacer(minLength: 8)
            MetricColumn(
                title: "DISCOVERED",
                value: "\(row.discoveryCount)",
                alignment: .trailing
            )
            .frame(minWidth: 44, alignment: .trailing)
        }
        .padding(10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: Theme.cardCornerRadius))
    }

    @ViewBuilder
    private func advisorImage(urlString: String?) -> some View {
        if let s = urlString, !s.isEmpty, let url = URL(string: s) {
            AsyncImage(url: url) { phase in
                switch phase {
                case .success(let image):
                    image.resizable().scaledToFit()
                default:
                    RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.2))
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8))
        } else {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.2))
                .overlay {
                    Image(systemName: "person.fill")
                        .foregroundStyle(Theme.secondaryText)
                }
        }
    }

    private func load() async {
        guard viewModel.hasSelectedFund else { return }
        let days = selectedLookbackDays
        isLoading = true
        do {
            let r = try await viewModel.fetchFundAdvisors(days: days)
            guard days == selectedLookbackDays else { return }
            advisors = r.advisors
            loadError = nil
        } catch {
            guard days == selectedLookbackDays else { return }
            loadError = error.localizedDescription
            advisors = []
            statsByAdvisorId = [:]
            isLoading = false
            return
        }
        isLoading = false

        if let s = try? await viewModel.fetchFundAdvisorScoreboard(days: days) {
            guard days == selectedLookbackDays else { return }
            statsByAdvisorId = Dictionary(uniqueKeysWithValues: s.advisors.map { ($0.advisorId, $0) })
        } else {
            guard days == selectedLookbackDays else { return }
            statsByAdvisorId = [:]
        }
    }

    private func cycleLookback() {
        guard let idx = lookbackOptions.firstIndex(of: selectedLookbackDays) else {
            selectedLookbackDays = lookbackOptions.first ?? 30
            return
        }
        let next = (idx + 1) % lookbackOptions.count
        selectedLookbackDays = lookbackOptions[next]
    }
}

/// Win rate 0…100% as a horizontal bar (30-day attributed BUYs from API scoreboard).
private struct AdvisorWinRateBar: View {
    let winRate: Double

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width * min(1, max(0, winRate / 100))
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.white.opacity(0.12))
                Capsule()
                    .fill(winRate >= 50 ? Color.green.opacity(0.75) : Color.orange.opacity(0.7))
                    .frame(width: max(3, w))
            }
        }
        .frame(height: 6)
    }
}

// MARK: - Discoveries for one advisor

struct AdvisorDiscoveriesView: View {
    @ObservedObject var viewModel: AuthViewModel
    @Binding var navigationPath: NavigationPath
    let advisorId: Int
    let advisorName: String
    let advisorDescription: String
    let advisorImageUrl: String?
    let advisorStats: AdvisorScoreboardRow?
    let lookbackDays: Int

    @Environment(\.dismiss) private var dismiss
    @State private var discoveries: [AdvisorDiscoveryRow] = []
    @State private var loadError: String?
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 10) {
            discoveriesHeaderCard
                .padding(.horizontal, 6)
                .padding(.top, 6)
            discoveriesSummaryCard
                .padding(.horizontal, 6)

            Group {
                if isLoading && discoveries.isEmpty && loadError == nil {
                    ProgressView().tint(.white)
                } else if let loadError {
                    Text(loadError)
                        .appStyle(.emptyStateMessage)
                        .padding()
                } else if discoveries.isEmpty {
                    Text("No discoveries yet.")
                        .appStyle(.emptyStateMessage)
                        .padding()
                } else {
                    List {
                        ForEach(discoveries) { row in
                            Button {
                                navigationPath.append(DiscoveryDetailNav(discoveryId: row.id))
                            } label: {
                                discoveryRow(row)
                            }
                            .buttonStyle(.plain)
                            .listRowInsets(EdgeInsets(top: 2, leading: 6, bottom: 4, trailing: 6))
                            .listRowBackground(Color.clear)
                            .listRowSeparator(.hidden)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.appBackground)
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbarBackground(Theme.appBackground, for: .navigationBar)
        .task(id: advisorId) { await load() }
    }

    private var discoveriesHeaderCard: some View {
        let desc = advisorDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.headline)
                        .foregroundStyle(.white)
                }
                .accessibilityLabel("Back")

                VStack(alignment: .leading, spacing: 2) {
                    Text(advisorName)
                        .appStyle(.screenHeadline)
                        .lineLimit(2)

                    if !desc.isEmpty {
                        Text(desc)
                            .appStyle(.screenSubline)
                            .lineLimit(4)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                Spacer(minLength: 4)

                headerAdvisorImage(urlString: advisorImageUrl)
                    .frame(width: 24, height: 24)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private var discoveriesSummaryCard: some View {
        let buys = advisorStats?.trades
        let winRate = advisorStats?.winRate
        let pnlPct = advisorStats?.gainLossPct
        return HStack(alignment: .top, spacing: 10) {
            MetricColumn(
                title: "FINDS",
                value: String(discoveries.count),
                expands: true
            )
            MetricColumn(
                title: "BUYS",
                value: buys.map(String.init) ?? "—",
                expands: true
            )
            MetricColumn(
                title: "WINS",
                value: winRate.map { String(format: "%.0f%%", $0) } ?? "—",
                expands: true
            )
            MetricColumn(
                title: "P&L %",
                value: pnlPct.map(formatPnlPercent) ?? "—",
                valueColor: Theme.signedColor(for: pnlPct),
                expands: true
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
    }


    private func formatPnlPercent(_ pct: Double) -> String {
        let sign = pct > 0 ? "+" : ""
        return "\(sign)\(String(format: "%.1f", pct))%"
    }

    @ViewBuilder
    private func headerAdvisorImage(urlString: String?) -> some View {
        if let s = urlString, !s.isEmpty, let url = URL(string: s) {
            AsyncImage(url: url) { phase in
                switch phase {
                case .success(let image):
                    image.resizable().scaledToFit()
                default:
                    RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
        } else {
            RoundedRectangle(cornerRadius: 5)
                .fill(Color.gray.opacity(0.15))
                .overlay {
                    Image(systemName: "person.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.secondaryText)
                }
        }
    }

    private func discoveryRow(_ row: AdvisorDiscoveryRow) -> some View {
        HStack(spacing: 12) {
            discoveryImageTicker(symbol: row.stock.symbol)
            VStack(alignment: .leading, spacing: 3) {
                Text(primaryTitle(for: row.stock))
                    .appStyle(.listHeadline)
                    .lineLimit(2)
                let expl = row.explanationLine.trimmingCharacters(in: .whitespacesAndNewlines)
                if !expl.isEmpty {
                    Text(expl)
                        .appStyle(.listSubline)
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Spacer(minLength: 4)
            healthScoreColumn(scoreText: row.listScoreText)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
    }

    private func primaryTitle(for stock: AdvisorDiscoveryStock) -> String {
        let c = stock.company.trimmingCharacters(in: .whitespacesAndNewlines)
        if !c.isEmpty { return c }
        return stock.symbol
    }

    private func discoveryImageTicker(symbol: String) -> some View {
        VStack(spacing: 4) {
            AsyncImage(url: URL(string: "https://images.financialmodelingprep.com/symbol/\(symbol).png")) { image in
                image.resizable().scaledToFit()
            } placeholder: {
                RoundedRectangle(cornerRadius: 5).fill(Color.gray.opacity(0.15))
            }
            .frame(width: 26, height: 26)

            Text(symbol)
                .appStyle(.tickerSymbol)
                .lineLimit(1)
        }
        .frame(width: 50, alignment: .leading)
    }

    private func healthScoreColumn(scoreText: String) -> some View {
        MetricColumn(title: "SCORE", value: scoreText, alignment: .trailing)
            .frame(minWidth: 56, alignment: .trailing)
    }

    private func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let r = try await viewModel.fetchAdvisorDiscoveries(advisorId: advisorId, days: lookbackDays)
            discoveries = r.discoveries
            loadError = nil
        } catch {
            loadError = error.localizedDescription
            discoveries = []
        }
    }
}

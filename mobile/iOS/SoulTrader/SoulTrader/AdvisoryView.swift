import SwiftUI

struct AdvisoryView: View {
    @ObservedObject var viewModel: AuthViewModel
    @State private var advisors: [FundAdvisorRow] = []
    @State private var loadError: String?
    @State private var isLoading = false

    var body: some View {
        NavigationStack {
            Group {
                if isLoading && advisors.isEmpty && loadError == nil {
                    ProgressView()
                        .tint(.white)
                } else if let loadError {
                    Text(loadError)
                        .font(.footnote)
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                } else if advisors.isEmpty {
                    Text("No advisors for this fund.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                } else {
                    List {
                        ForEach(advisors) { row in
                            NavigationLink(
                                value: AdvisorNav(
                                    id: row.id,
                                    name: row.name,
                                    description: row.description,
                                    imageUrl: row.imageUrl
                                )
                            ) {
                                advisorRow(row)
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
                    advisorId: nav.id,
                    advisorName: nav.name,
                    advisorDescription: nav.description,
                    advisorImageUrl: nav.imageUrl
                )
            }
            .navigationDestination(for: DiscoveryDetailNav.self) { nav in
                DiscoveryDetailView(
                    discoveryId: nav.discoveryId,
                    baseURL: viewModel.selectedHost.baseURL,
                    viewModel: viewModel
                )
            }
        }
        .task(id: viewModel.selectedFundId) { await load() }
        .onChange(of: viewModel.selectedTab) { _, tab in
            if tab == .advisory { Task { await load() } }
        }
    }

    private func advisorRow(_ row: FundAdvisorRow) -> some View {
        HStack(alignment: .top, spacing: 12) {
            advisorImage(urlString: row.imageUrl)
                .frame(width: 44, height: 44)
            VStack(alignment: .leading, spacing: 4) {
                Text(row.name)
                    .font(.headline)
                    .foregroundStyle(Theme.valuePrimary)
                if !row.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text(row.description.trimmingCharacters(in: .whitespacesAndNewlines))
                        .font(.subheadline)
                        .foregroundStyle(Theme.secondaryText)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 2) {
                Text("STOCKS")
                    .font(.caption2)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.labelAccent)
                Text("\(row.discoveryCount)")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.valuePrimary)
            }
            .frame(minWidth: 44, alignment: .trailing)
        }
        .padding(10)
        .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))
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
        isLoading = true
        defer { isLoading = false }
        do {
            let r = try await viewModel.fetchFundAdvisors()
            advisors = r.advisors
            loadError = nil
        } catch {
            loadError = error.localizedDescription
            advisors = []
        }
    }
}

// MARK: - Discoveries for one advisor

struct AdvisorDiscoveriesView: View {
    @ObservedObject var viewModel: AuthViewModel
    let advisorId: Int
    let advisorName: String
    let advisorDescription: String
    let advisorImageUrl: String?

    @Environment(\.dismiss) private var dismiss
    @State private var discoveries: [AdvisorDiscoveryRow] = []
    @State private var loadError: String?
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 10) {
            discoveriesHeaderCard
                .padding(.horizontal, 6)
                .padding(.top, 6)

            Group {
                if isLoading && discoveries.isEmpty && loadError == nil {
                    ProgressView().tint(.white)
                } else if let loadError {
                    Text(loadError)
                        .font(.footnote)
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                } else if discoveries.isEmpty {
                    Text("No discoveries yet.")
                        .font(.subheadline)
                        .foregroundStyle(Theme.secondaryText)
                        .padding()
                } else {
                    List {
                        ForEach(discoveries) { row in
                            NavigationLink(value: DiscoveryDetailNav(discoveryId: row.id)) {
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
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .lineLimit(2)

                    if !desc.isEmpty {
                        Text(desc)
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundStyle(Theme.secondaryText)
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
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(Theme.valuePrimary)
                    .lineLimit(2)
                let expl = row.explanationLine.trimmingCharacters(in: .whitespacesAndNewlines)
                if !expl.isEmpty {
                    Text(expl)
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryText)
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Spacer(minLength: 4)
            healthScoreColumn(health: row.healthScore)
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
                .font(.system(size: 11, weight: .bold))
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
        }
        .frame(width: 50, alignment: .leading)
    }

    private func healthScoreColumn(health: Double?) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text("SCORE")
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)
            Text(formatOptionalScore(health))
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.valuePrimary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(minWidth: 56, alignment: .trailing)
    }

    private func formatOptionalScore(_ v: Double?) -> String {
        guard let v else { return "—" }
        if abs(v) < 1e-9 {
            return "AVOID"
        }
        return String(format: "%.1f", v)
    }

    private func load() async {
        isLoading = true
        defer { isLoading = false }
        do {
            let r = try await viewModel.fetchAdvisorDiscoveries(advisorId: advisorId)
            discoveries = r.discoveries
            loadError = nil
        } catch {
            loadError = error.localizedDescription
            discoveries = []
        }
    }
}

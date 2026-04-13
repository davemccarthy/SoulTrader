import SwiftUI

struct LoginView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                BrandHeaderTitleView(title: "SOULTRADER")
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(
                LinearGradient(
                    colors: [Theme.brandHeaderStart, Theme.brandHeaderEnd],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )

            ScrollView {
                VStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 10) {
                        TextField(
                            "",
                            text: $viewModel.username,
                            prompt: Text("Username").foregroundStyle(Theme.secondaryText)
                        )
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .foregroundStyle(Theme.valuePrimary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 10)
                            .background(Color.black.opacity(0.26), in: RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.white.opacity(0.10), lineWidth: 1)
                            )
                            .tint(Theme.valuePrimary)

                        SecureField(
                            "",
                            text: $viewModel.password,
                            prompt: Text("Password").foregroundStyle(Theme.secondaryText)
                        )
                            .foregroundStyle(Theme.valuePrimary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 10)
                            .background(Color.black.opacity(0.26), in: RoundedRectangle(cornerRadius: 8))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.white.opacity(0.10), lineWidth: 1)
                            )
                            .tint(Theme.valuePrimary)

                        Picker("Host", selection: $viewModel.selectedHost) {
                            ForEach(APIEnvironment.HostOption.allCases) { host in
                                Text(host.rawValue).tag(host)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .tint(Theme.valuePrimary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.26), in: RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.white.opacity(0.10), lineWidth: 1)
                        )
                    }
                    .padding(.vertical, 10)
                    .padding(.horizontal, 10)
                    .background(Theme.rowBackground, in: RoundedRectangle(cornerRadius: 10))

                    Button {
                        Task { await viewModel.login() }
                    } label: {
                        Text("Login")
                            .font(.headline)
                            .fontWeight(.semibold)
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .background(Theme.brandHeaderStart, in: RoundedRectangle(cornerRadius: 10))
                    .disabled(viewModel.isLoading || viewModel.username.isEmpty || viewModel.password.isEmpty)
                    .opacity(viewModel.isLoading || viewModel.username.isEmpty || viewModel.password.isEmpty ? 0.65 : 1.0)
                }
                .padding(.horizontal, 6)
                .padding(.top, 8)
                .padding(.bottom, 12)
            }
            .scrollContentBackground(.hidden)
            .background(Theme.appBackground)
        }
        .background(Theme.appBackground)
    }
}

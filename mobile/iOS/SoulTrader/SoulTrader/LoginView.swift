import SwiftUI
import UIKit

struct LoginView: View {
    @ObservedObject var viewModel: AuthViewModel
    @FocusState private var focusedField: LoginFocusedField?

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
                    Group {
                        if UIImage(named: "LoginHero") != nil {
                            Image("LoginHero")
                                .resizable()
                                .renderingMode(.original)
                                .scaledToFit()
                                .frame(maxHeight: 200)
                                .padding(.horizontal, 20)
                                .padding(.top, 16)
                                .accessibilityLabel("SoulTrader")
                        }
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        CredentialFieldBlock(
                            caption: "USERNAME",
                            systemImage: "person.fill",
                            focused: focusedField == .username
                        ) {
                            TextField(
                                "",
                                text: $viewModel.username,
                                prompt: Text("Enter username").foregroundStyle(Theme.secondaryText)
                            )
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .foregroundStyle(Theme.valuePrimary)
                            .focused($focusedField, equals: .username)
                            .submitLabel(.next)
                            .onSubmit { focusedField = .password }
                            .tint(Theme.valuePrimary)
                        }

                        CredentialFieldBlock(
                            caption: "PASSWORD",
                            systemImage: "lock.fill",
                            focused: focusedField == .password
                        ) {
                            SecureField(
                                "",
                                text: $viewModel.password,
                                prompt: Text("Enter password").foregroundStyle(Theme.secondaryText)
                            )
                            .foregroundStyle(Theme.valuePrimary)
                            .focused($focusedField, equals: .password)
                            .submitLabel(.go)
                            .onSubmit {
                                if !viewModel.username.isEmpty, !viewModel.password.isEmpty {
                                    Task { await viewModel.login() }
                                }
                            }
                            .tint(Theme.valuePrimary)
                        }

                        if APIEnvironment.showHostCredential {
                            CredentialFieldBlock(
                                caption: "HOST",
                                systemImage: "network",
                                focused: false
                            ) {
                                Menu {
                                    ForEach(APIEnvironment.HostOption.allCases) { host in
                                        Button {
                                            viewModel.selectedHost = host
                                        } label: {
                                            HStack {
                                                Text(host.rawValue)
                                                if host == viewModel.selectedHost {
                                                    Image(systemName: "checkmark")
                                                }
                                            }
                                        }
                                    }
                                } label: {
                                    HStack {
                                        Text(viewModel.selectedHost.rawValue)
                                            .font(.body)
                                            .foregroundStyle(Theme.valuePrimary)
                                            .lineLimit(1)
                                        Spacer(minLength: 8)
                                        Image(systemName: "chevron.up.chevron.down")
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(Theme.secondaryText)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .contentShape(Rectangle())
                                }
                                .tint(Theme.valuePrimary)
                            }
                        }
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

private enum LoginFocusedField: Hashable {
    case username
    case password
}

private struct CredentialFieldBlock<Content: View>: View {
    let caption: String
    let systemImage: String
    var focused: Bool
    @ViewBuilder var content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(caption)
                .font(.caption2)
                .fontWeight(.semibold)
                .foregroundStyle(Theme.labelAccent)

            HStack(alignment: .center, spacing: 10) {
                Image(systemName: systemImage)
                    .font(.body)
                    .foregroundStyle(Theme.labelAccent.opacity(0.92))
                    .frame(width: 22, alignment: .center)
                    .accessibilityHidden(true)

                content()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.26), in: RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(borderColor, lineWidth: focused ? 1.5 : 1)
            )
        }
    }

    private var borderColor: Color {
        if focused {
            return Theme.brandHeaderEnd.opacity(0.65)
        }
        return Color.white.opacity(0.10)
    }
}

import SwiftUI

struct LoginView: View {
    @ObservedObject var viewModel: AuthViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Login") {
                    Picker("Host", selection: $viewModel.selectedHost) {
                        ForEach(APIEnvironment.HostOption.allCases) { host in
                            Text(host.rawValue).tag(host)
                        }
                    }
                    TextField("Username", text: $viewModel.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    SecureField("Password", text: $viewModel.password)
                }

                Section {
                    Button("Login") {
                        Task { await viewModel.login() }
                    }
                    .disabled(viewModel.isLoading || viewModel.username.isEmpty || viewModel.password.isEmpty)
                }

                if let statusMessage = viewModel.statusMessage {
                    Section("Status") {
                        Text(statusMessage).font(.footnote)
                    }
                }
            }
            .navigationTitle("SoulTrader")
        }
    }
}

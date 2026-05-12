//
//  AppDelegate.swift
//  SoulTrader
//

import UIKit
import UserNotifications
import FirebaseCore
import FirebaseMessaging

extension Notification.Name {
    /// Posted when FCM registration token is created or refreshed (`object` is `String` token).
    static let soultraderFCMTokenDidChange = Notification.Name("soultraderFCMTokenDidChange")
}

/// `@objcMembers` so Firebase’s AppDelegate swizzler / ObjC runtime see `UIApplicationDelegate` hooks reliably with SwiftUI.
@objcMembers
final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate, MessagingDelegate {

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        // Must run after `UIApplication.shared.delegate` is this object (SwiftUI installs the adaptor before this runs).
        // Calling `configure()` from `App.init()` is too early and breaks Firebase’s delegate discovery (I-SWZ001014).
        FirebaseApp.configure()

        UNUserNotificationCenter.current().delegate = self
        Messaging.messaging().delegate = self

        Task {
            let center = UNUserNotificationCenter.current()
            do {
                let granted = try await center.requestAuthorization(options: [.alert, .sound, .badge])
                if granted {
                    await MainActor.run {
                        application.registerForRemoteNotifications()
                    }
                }
            } catch {
                // User denied or request failed; FCM token may still be unavailable until APNs succeeds.
            }
        }

        return true
    }

    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        Messaging.messaging().apnsToken = deviceToken
    }

    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        // Common on Simulator; real devices need Push capability + provisioning.
    }

    // MARK: - MessagingDelegate

    func messaging(_ messaging: Messaging, didReceiveRegistrationToken fcmToken: String?) {
        guard let fcmToken, !fcmToken.isEmpty else { return }
        NotificationCenter.default.post(name: .soultraderFCMTokenDidChange, object: fcmToken)
    }

    // MARK: - UNUserNotificationCenterDelegate

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound, .badge])
    }
}

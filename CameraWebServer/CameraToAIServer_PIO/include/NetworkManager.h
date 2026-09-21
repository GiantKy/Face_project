#ifndef NETWORK_MANAGER_H
#define NETWORK_MANAGER_H

#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>

/**
 * @class NetworkManager
 * @brief Quản lý toàn bộ kết nối WiFi và dịch vụ mDNS.
 */
class NetworkManager {
public:
    NetworkManager(const char* ssid, const char* password)
        : m_ssid(ssid), m_password(password) {}

    bool connect(int maxRetries = 35) {
        Serial.printf("[NetworkManager] Dang ket noi toi WiFi: %s ", m_ssid);
        WiFi.disconnect(true);
        delay(100);
        WiFi.mode(WIFI_STA);
        WiFi.setSleep(false);
        WiFi.begin(m_ssid, m_password);

        int retry = 0;
        while (WiFi.status() != WL_CONNECTED && retry < maxRetries) {
            delay(500);
            Serial.print(".");
            retry++;
        }

        if (WiFi.status() == WL_CONNECTED) {
            Serial.println("\n[NetworkManager] Ket noi WiFi thanh cong!");
            Serial.print("[NetworkManager] Dia chi IP ESP32: http://");
            Serial.println(WiFi.localIP());

            if (MDNS.begin("esp32cam")) {
                Serial.println("[NetworkManager] mDNS san sang: http://esp32cam.local");
            }
            return true;
        } else {
            Serial.println("\n[NetworkManager] Ket noi WiFi that bai! Vui long kiem tra lai SSID va Password.");
            return false;
        }
    }

    bool isConnected() const {
        return (WiFi.status() == WL_CONNECTED);
    }

    IPAddress getLocalIP() const {
        return WiFi.localIP();
    }

private:
    const char* m_ssid;
    const char* m_password;
};

#endif // NETWORK_MANAGER_H
